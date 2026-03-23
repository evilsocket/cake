use std::io;
use std::time::Instant;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use ratatui::{
    layout::{Constraint, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState,
    },
    Frame, Terminal,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── Data types ─────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    Cluster,
}

// ── Topology API response types ────────────────────────────────────────────

#[derive(Deserialize, Clone, Default)]
struct TopologyResponse {
    #[serde(default)]
    model: String,
    #[serde(default)]
    model_id: String,
    #[serde(default)]
    dtype: String,
    #[serde(default)]
    num_layers: usize,
    #[serde(default)]
    memory_bytes: u64,
    #[serde(default)]
    master: MasterInfo,
    #[serde(default)]
    workers: Vec<WorkerInfo>,
}

#[derive(Deserialize, Clone, Default)]
struct MasterInfo {
    #[serde(default)]
    backend: String,
    #[serde(default)]
    layers: Vec<String>,
    #[serde(default)]
    vram_bytes: u64,
    #[serde(default)]
    tflops: f64,
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    os: String,
}

#[derive(Deserialize, Clone, Default)]
struct WorkerInfo {
    #[serde(default)]
    name: String,
    #[serde(default)]
    layers: Vec<String>,
    #[serde(default)]
    vram_bytes: u64,
    #[serde(default)]
    tflops: f64,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    os: String,
}

// ── SSE response parsing types ─────────────────────────────────────────────

#[derive(Deserialize)]
struct StreamResponse {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

// ── App state ──────────────────────────────────────────────────────────────

struct App {
    // Tab
    tab: Tab,
    // Chat state
    messages: Vec<ChatMessage>,
    input: String,
    cursor_pos: usize,
    scroll_offset: u16,
    total_lines: u16,
    streaming: bool,
    // Cluster state
    topology: Option<TopologyResponse>,
    cluster_scroll: u16,
    cluster_total_lines: u16,
    // Common
    server: String,
    status: String,
    should_quit: bool,
    auto_scroll: bool,
}

impl App {
    fn new(server: &str) -> Self {
        Self {
            tab: Tab::Chat,
            messages: Vec::new(),
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            total_lines: 0,
            streaming: false,
            topology: None,
            cluster_scroll: 0,
            cluster_total_lines: 0,
            server: server.to_string(),
            status: String::new(),
            should_quit: false,
            auto_scroll: true,
        }
    }

    fn scroll_to_bottom(&mut self, visible_height: u16) {
        if self.total_lines > visible_height {
            self.scroll_offset = self.total_lines - visible_height;
        } else {
            self.scroll_offset = 0;
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn human_bytes(bytes: u64) -> String {
    if bytes == 0 {
        return "0 B".to_string();
    }
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit_idx = 0;
    while value >= 1024.0 && unit_idx < units.len() - 1 {
        value /= 1024.0;
        unit_idx += 1;
    }
    if unit_idx == 0 {
        format!("{} B", bytes)
    } else {
        format!("{:.1} {}", value, units[unit_idx])
    }
}

fn build_message_lines(messages: &[ChatMessage], width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let wrap_width = if width > 4 {
        width as usize - 4
    } else {
        width as usize
    };

    for (i, msg) in messages.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }

        let (prefix, style) = if msg.role == "user" {
            (
                "you> ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ", Style::default().fg(Color::White))
        };

        for text_line in msg.content.split('\n') {
            let full = format!("{}{}", prefix, text_line);
            if wrap_width == 0 {
                lines.push(Line::from(Span::styled(full, style)));
                continue;
            }
            let chars: Vec<char> = full.chars().collect();
            if chars.is_empty() {
                lines.push(Line::from(Span::styled(String::new(), style)));
            } else {
                let mut pos = 0;
                while pos < chars.len() {
                    let end = (pos + wrap_width).min(chars.len());
                    let chunk: String = chars[pos..end].iter().collect();
                    lines.push(Line::from(Span::styled(chunk, style)));
                    pos = end;
                }
            }
        }
    }

    lines
}

// ── Drawing ────────────────────────────────────────────────────────────────

fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();

    let layout = Layout::vertical([
        Constraint::Length(1), // tab bar
        Constraint::Min(1),   // content
    ])
    .split(area);

    draw_tab_bar(frame, app, layout[0]);

    match app.tab {
        Tab::Chat => draw_chat(frame, app, layout[1]),
        Tab::Cluster => draw_cluster(frame, app, layout[1]),
    }
}

fn draw_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let chat_style = if app.tab == Tab::Chat {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let cluster_style = if app.tab == Tab::Cluster {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let mut spans = vec![
        Span::raw(" "),
        Span::styled("[1] Chat", chat_style),
        Span::raw("  "),
        Span::styled("[2] Cluster", cluster_style),
        Span::raw("  "),
        Span::styled(&app.server, Style::default().fg(Color::DarkGray)),
    ];

    if !app.status.is_empty() {
        spans.push(Span::raw(" — "));
        spans.push(Span::styled(
            app.status.clone(),
            Style::default().fg(Color::Green),
        ));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_chat(frame: &mut Frame, app: &mut App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Min(1),   // messages
        Constraint::Length(3), // input
    ])
    .split(area);

    // Messages area
    let msg_area = layout[0];
    let inner_width = msg_area.width.saturating_sub(2);
    let visible_height = msg_area.height.saturating_sub(2);

    let all_lines = build_message_lines(&app.messages, inner_width);
    app.total_lines = all_lines.len() as u16;

    if app.auto_scroll {
        app.scroll_to_bottom(visible_height);
    }

    let msg_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(if app.streaming {
            " generating... "
        } else {
            ""
        });

    let messages_widget = Paragraph::new(all_lines)
        .block(msg_block)
        .scroll((app.scroll_offset, 0));

    frame.render_widget(messages_widget, msg_area);

    if app.total_lines > visible_height {
        let mut scrollbar_state = ScrollbarState::new(app.total_lines as usize)
            .position(app.scroll_offset as usize)
            .viewport_content_length(visible_height as usize);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            msg_area.inner(Margin::new(0, 1)),
            &mut scrollbar_state,
        );
    }

    // Input area
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(if app.streaming {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::Cyan)
        })
        .title(" message ");

    let input_widget = Paragraph::new(app.input.as_str()).block(input_block);
    frame.render_widget(input_widget, layout[1]);

    if !app.streaming {
        frame.set_cursor_position((
            layout[1].x + 1 + app.cursor_pos as u16,
            layout[1].y + 1,
        ));
    }
}

fn draw_cluster(frame: &mut Frame, app: &mut App, area: Rect) {
    let topo = match &app.topology {
        Some(t) => t.clone(),
        None => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" cluster ");
            let msg = Paragraph::new("  loading topology...")
                .style(Style::default().fg(Color::DarkGray))
                .block(block);
            frame.render_widget(msg, area);
            return;
        }
    };

    // Build all node info into a flat list of lines for scrollable rendering
    let inner_width = area.width.saturating_sub(2) as usize;
    let visible_height = area.height.saturating_sub(2);

    let lines = build_cluster_lines(&topo, inner_width);
    app.cluster_total_lines = lines.len() as u16;

    // Clamp scroll
    if app.cluster_total_lines > visible_height {
        app.cluster_scroll = app
            .cluster_scroll
            .min(app.cluster_total_lines - visible_height);
    } else {
        app.cluster_scroll = 0;
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(format!(
            " {} — {} layers — {} ",
            &topo.model, topo.num_layers, &topo.dtype
        ));

    let cluster_widget = Paragraph::new(lines)
        .block(block)
        .scroll((app.cluster_scroll, 0));

    frame.render_widget(cluster_widget, area);

    if app.cluster_total_lines > visible_height {
        let mut scrollbar_state = ScrollbarState::new(app.cluster_total_lines as usize)
            .position(app.cluster_scroll as usize)
            .viewport_content_length(visible_height as usize);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area.inner(Margin::new(0, 1)),
            &mut scrollbar_state,
        );
    }
}

fn build_cluster_lines(topo: &TopologyResponse, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Collect all nodes for VRAM bar scaling
    let all_vram: Vec<u64> = std::iter::once(topo.master.vram_bytes)
        .chain(topo.workers.iter().map(|w| w.vram_bytes))
        .collect();
    let max_vram = all_vram.iter().copied().max().unwrap_or(1).max(1);
    let bar_max_width = width.saturating_sub(30).min(40);

    // ── Model info ──
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  Model: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            topo.model_id.clone(),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Layers: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{}", topo.num_layers),
            Style::default().fg(Color::White),
        ),
        Span::styled("  dtype: ", Style::default().fg(Color::DarkGray)),
        Span::styled(topo.dtype.clone(), Style::default().fg(Color::White)),
        Span::styled("  memory: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            human_bytes(topo.memory_bytes),
            Style::default().fg(Color::White),
        ),
    ]));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {}", "-".repeat(width.saturating_sub(4))),
        Style::default().fg(Color::DarkGray),
    )));

    // ── Master node ──
    let master_label = if topo.master.hostname.is_empty() {
        "master".to_string()
    } else {
        topo.master.hostname.clone()
    };
    render_node(
        &mut lines,
        &master_label,
        &topo.master.backend,
        &topo.master.os,
        topo.master.vram_bytes,
        topo.master.tflops,
        &topo.master.layers,
        max_vram,
        bar_max_width,
        Color::Yellow,
        true,
    );

    // ── Worker nodes ──
    for w in &topo.workers {
        let label = if w.hostname.is_empty() {
            w.name.clone()
        } else {
            w.hostname.clone()
        };
        render_node(
            &mut lines,
            &label,
            &w.backend,
            &w.os,
            w.vram_bytes,
            w.tflops,
            &w.layers,
            max_vram,
            bar_max_width,
            Color::Cyan,
            false,
        );
    }

    // ── Layer distribution summary ──
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  {}", "-".repeat(width.saturating_sub(4))),
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Layer Distribution",
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    // Master layers
    let master_name = if topo.master.hostname.is_empty() {
        "master"
    } else {
        &topo.master.hostname
    };
    push_layer_bar(
        &mut lines,
        master_name,
        topo.master.layers.len(),
        topo.num_layers,
        bar_max_width,
        Color::Yellow,
    );

    // Worker layers
    for w in &topo.workers {
        let name = if w.hostname.is_empty() {
            &w.name
        } else {
            &w.hostname
        };
        push_layer_bar(
            &mut lines,
            name,
            w.layers.len(),
            topo.num_layers,
            bar_max_width,
            Color::Cyan,
        );
    }

    lines.push(Line::from(""));

    lines
}

#[allow(clippy::too_many_arguments)]
fn render_node(
    lines: &mut Vec<Line<'static>>,
    name: &str,
    backend: &str,
    os: &str,
    vram_bytes: u64,
    tflops: f64,
    layers: &[String],
    max_vram: u64,
    bar_max_width: usize,
    color: Color,
    is_master: bool,
) {
    lines.push(Line::from(""));

    // Node header
    let role = if is_master { " (master)" } else { "" };
    lines.push(Line::from(vec![
        Span::styled(
            format!("  {}{}", name, role),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
    ]));

    // Backend + OS
    let mut info_spans = vec![Span::raw("  ")];
    if !backend.is_empty() {
        info_spans.push(Span::styled(
            backend.to_string(),
            Style::default().fg(Color::White),
        ));
    }
    if !os.is_empty() {
        if !backend.is_empty() {
            info_spans.push(Span::styled(" / ", Style::default().fg(Color::DarkGray)));
        }
        info_spans.push(Span::styled(
            os.to_string(),
            Style::default().fg(Color::DarkGray),
        ));
    }
    if !info_spans.is_empty() {
        lines.push(Line::from(info_spans));
    }

    // VRAM bar
    if vram_bytes > 0 {
        let bar_filled = ((vram_bytes as f64 / max_vram as f64) * bar_max_width as f64) as usize;
        let bar_filled = bar_filled.max(1).min(bar_max_width);
        let bar_empty = bar_max_width.saturating_sub(bar_filled);
        lines.push(Line::from(vec![
            Span::styled("  VRAM ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "\u{2588}".repeat(bar_filled),
                Style::default().fg(color),
            ),
            Span::styled(
                "\u{2591}".repeat(bar_empty),
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" "),
            Span::styled(
                human_bytes(vram_bytes),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    // TFLOPS
    if tflops > 0.0 {
        lines.push(Line::from(vec![
            Span::styled("  TFLOPS ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}", tflops),
                Style::default().fg(Color::White),
            ),
        ]));
    }

    // Layers
    if !layers.is_empty() {
        let range = format!(
            "{} - {} ({} layers)",
            layers.first().unwrap_or(&String::new()),
            layers.last().unwrap_or(&String::new()),
            layers.len()
        );
        lines.push(Line::from(vec![
            Span::styled("  Layers ", Style::default().fg(Color::DarkGray)),
            Span::styled(range, Style::default().fg(Color::White)),
        ]));
    }
}

fn push_layer_bar(
    lines: &mut Vec<Line<'static>>,
    name: &str,
    count: usize,
    total: usize,
    bar_max_width: usize,
    color: Color,
) {
    let pct = if total > 0 {
        count as f64 / total as f64
    } else {
        0.0
    };
    let bar_filled = (pct * bar_max_width as f64) as usize;
    let bar_filled = if count > 0 { bar_filled.max(1) } else { 0 };
    let bar_empty = bar_max_width.saturating_sub(bar_filled);

    lines.push(Line::from(vec![
        Span::styled(
            format!("  {:>12} ", name),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            "\u{2588}".repeat(bar_filled),
            Style::default().fg(color),
        ),
        Span::styled(
            "\u{2591}".repeat(bar_empty),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw(" "),
        Span::styled(
            format!("{} ({:.0}%)", count, pct * 100.0),
            Style::default().fg(Color::White),
        ),
    ]));
}

// ── SSE streaming ──────────────────────────────────────────────────────────

async fn stream_response(
    client: &Client,
    server: &str,
    messages: &[ChatMessage],
) -> Result<tokio::sync::mpsc::UnboundedReceiver<Option<String>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let body = serde_json::json!({
        "messages": messages,
        "stream": true
    });

    // Use a blocking thread for the HTTP stream to avoid tokio runtime conflicts
    // when the server runs on a separate runtime (local chat mode).
    let url = format!("{}/v1/chat/completions", server);
    let body_str = serde_json::to_string(&body)?;

    std::thread::spawn(move || {
        let resp = match reqwest::blocking::Client::new()
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body_str)
            .send()
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Some(format!("[connection error: {}]", e)));
                let _ = tx.send(None);
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = tx.send(Some(format!("[API error: {}]", resp.status())));
            let _ = tx.send(None);
            return;
        }

        let reader = std::io::BufReader::new(resp);
        use std::io::BufRead;
        let mut buffer = String::new();

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => break,
            };

            if line.is_empty() {
                // Process buffered event
                for event_line in buffer.lines() {
                    let event_line = event_line.trim();
                    if event_line == "data: [DONE]" {
                        let _ = tx.send(None);
                        return;
                    }
                    if let Some(data) = event_line.strip_prefix("data: ") {
                        if let Ok(resp) = serde_json::from_str::<StreamResponse>(data) {
                            if let Some(choice) = resp.choices.first() {
                                if let Some(ref content) = choice.delta.content {
                                    let _ = tx.send(Some(content.clone()));
                                }
                                if choice.finish_reason.is_some() {
                                    let _ = tx.send(None);
                                    return;
                                }
                            }
                        }
                    }
                }
                buffer.clear();
            } else {
                buffer.push_str(&line);
                buffer.push('\n');
            }
        }

        let _ = tx.send(None);
    });

    Ok(rx)
}

async fn fetch_topology(client: &Client, server: &str) -> Result<TopologyResponse> {
    let resp = client
        .get(format!("{}/api/v1/topology", server))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("topology API error: {}", resp.status());
    }
    Ok(resp.json().await?)
}

// ── Main loop ──────────────────────────────────────────────────────────────

/// Remote chat: connect to an API server.
pub async fn run_remote(server: &str) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal, server).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    server: &str,
) -> Result<()> {
    let mut app = App::new(server);
    let client = Client::new();
    let mut response_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Option<String>>> = None;
    let mut gen_start: Option<Instant> = None;
    let mut token_count: usize = 0;
    let mut last_topology_fetch = Instant::now() - std::time::Duration::from_secs(60);

    loop {
        // Fetch topology periodically (every 10s) or on first switch to cluster tab
        if app.tab == Tab::Cluster
            && last_topology_fetch.elapsed() > std::time::Duration::from_secs(10)
        {
            if let Ok(topo) = fetch_topology(&client, server).await {
                app.topology = Some(topo);
            }
            last_topology_fetch = Instant::now();
        }

        terminal.draw(|f| draw(f, &mut app))?;

        if app.should_quit {
            break;
        }

        let timeout = std::time::Duration::from_millis(16);

        tokio::select! {
            _ = tokio::time::sleep(timeout) => {
                while event::poll(std::time::Duration::ZERO)? {
                    if let Event::Key(key) = event::read()? {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        match key.code {
                            KeyCode::Esc => {
                                app.should_quit = true;
                            }
                            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                app.should_quit = true;
                            }
                            // Tab switching
                            KeyCode::Char('1') if !app.streaming => {
                                app.tab = Tab::Chat;
                            }
                            KeyCode::Char('2') if !app.streaming => {
                                app.tab = Tab::Cluster;
                                // Force immediate fetch
                                if app.topology.is_none() {
                                    last_topology_fetch = Instant::now() - std::time::Duration::from_secs(60);
                                }
                            }
                            KeyCode::Tab if !app.streaming => {
                                app.tab = match app.tab {
                                    Tab::Chat => Tab::Cluster,
                                    Tab::Cluster => Tab::Chat,
                                };
                                if app.tab == Tab::Cluster && app.topology.is_none() {
                                    last_topology_fetch = Instant::now() - std::time::Duration::from_secs(60);
                                }
                            }
                            // Chat-specific keys
                            KeyCode::Enter if app.tab == Tab::Chat && !app.streaming && !app.input.is_empty() => {
                                let user_msg = ChatMessage {
                                    role: "user".to_string(),
                                    content: app.input.clone(),
                                };
                                app.messages.push(user_msg);
                                app.input.clear();
                                app.cursor_pos = 0;
                                app.auto_scroll = true;

                                app.messages.push(ChatMessage {
                                    role: "assistant".to_string(),
                                    content: String::new(),
                                });
                                app.streaming = true;
                                app.status = "thinking...".to_string();
                                gen_start = Some(Instant::now());
                                token_count = 0;

                                app.status = format!("connecting to {} ...", server);
                                match stream_response(&client, server, &app.messages[..app.messages.len()-1]).await {
                                    Ok(rx) => {
                                        app.status = "streaming...".to_string();
                                        response_rx = Some(rx);
                                    }
                                    Err(e) => {
                                        if let Some(last) = app.messages.last_mut() {
                                            last.content = format!("[error: {}]", e);
                                        }
                                        app.streaming = false;
                                        app.status = format!("error: {}", e);
                                        response_rx = None;
                                    }
                                }
                            }
                            KeyCode::Char(c) if app.tab == Tab::Chat && !app.streaming => {
                                app.input.insert(app.cursor_pos, c);
                                app.cursor_pos += 1;
                            }
                            KeyCode::Backspace if app.tab == Tab::Chat && !app.streaming && app.cursor_pos > 0 => {
                                app.cursor_pos -= 1;
                                app.input.remove(app.cursor_pos);
                            }
                            KeyCode::Delete if app.tab == Tab::Chat && !app.streaming && app.cursor_pos < app.input.len() => {
                                app.input.remove(app.cursor_pos);
                            }
                            KeyCode::Left if app.tab == Tab::Chat && app.cursor_pos > 0 => {
                                app.cursor_pos -= 1;
                            }
                            KeyCode::Right if app.tab == Tab::Chat && app.cursor_pos < app.input.len() => {
                                app.cursor_pos += 1;
                            }
                            KeyCode::Home if app.tab == Tab::Chat => {
                                app.cursor_pos = 0;
                            }
                            KeyCode::End if app.tab == Tab::Chat => {
                                app.cursor_pos = app.input.len();
                            }
                            // Scrolling (works in both tabs)
                            KeyCode::Up => {
                                match app.tab {
                                    Tab::Chat => {
                                        app.auto_scroll = false;
                                        app.scroll_offset = app.scroll_offset.saturating_sub(1);
                                    }
                                    Tab::Cluster => {
                                        app.cluster_scroll = app.cluster_scroll.saturating_sub(1);
                                    }
                                }
                            }
                            KeyCode::Down => {
                                match app.tab {
                                    Tab::Chat => {
                                        app.scroll_offset = app.scroll_offset.saturating_add(1);
                                    }
                                    Tab::Cluster => {
                                        app.cluster_scroll = app.cluster_scroll.saturating_add(1);
                                    }
                                }
                            }
                            KeyCode::PageUp => {
                                match app.tab {
                                    Tab::Chat => {
                                        app.auto_scroll = false;
                                        app.scroll_offset = app.scroll_offset.saturating_sub(10);
                                    }
                                    Tab::Cluster => {
                                        app.cluster_scroll = app.cluster_scroll.saturating_sub(10);
                                    }
                                }
                            }
                            KeyCode::PageDown => {
                                match app.tab {
                                    Tab::Chat => {
                                        app.scroll_offset = app.scroll_offset.saturating_add(10);
                                    }
                                    Tab::Cluster => {
                                        app.cluster_scroll = app.cluster_scroll.saturating_add(10);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            msg = async {
                if let Some(ref mut rx) = response_rx {
                    rx.recv().await
                } else {
                    std::future::pending::<Option<Option<String>>>().await
                }
            } => {
                match msg {
                    Some(Some(content)) => {
                        token_count += 1;
                        if let Some(last) = app.messages.last_mut() {
                            last.content.push_str(&content);
                        }
                        if let Some(start) = gen_start {
                            let elapsed = start.elapsed().as_secs_f64();
                            if elapsed > 0.0 {
                                app.status = format!("{:.1} tok/s", token_count as f64 / elapsed);
                            }
                        }
                        app.auto_scroll = true;
                    }
                    Some(None) | None => {
                        app.streaming = false;
                        response_rx = None;
                        if let Some(start) = gen_start.take() {
                            let elapsed = start.elapsed().as_secs_f64();
                            if elapsed > 0.0 {
                                app.status = format!(
                                    "{} tokens in {:.1}s ({:.1} tok/s)",
                                    token_count,
                                    elapsed,
                                    token_count as f64 / elapsed
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}


// ─── Local chat mode ─────────────────────────────────────────────────

use cake_core::cake::Context;

/// Local chat: start an ephemeral API server, then connect the TUI to it.
/// This reuses the existing remote chat TUI with zero code duplication.
pub async fn run_local(ctx: &mut Context) -> Result<()> {
    // Find a free port by temporarily binding to :0
    let tmp = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = tmp.local_addr()?.port();
    drop(tmp);

    let addr = format!("127.0.0.1:{port}");
    let server_url = format!("http://{addr}");

    eprintln!("starting local chat on {server_url}");
    eprintln!("loading model (logs visible until TUI starts)...\n");

    ctx.args.api = Some(addr);

    // Start the master on a background thread. Model loading logs are visible
    // on the terminal until the TUI takes over.
    let ctx_clone = ctx.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            if let Err(e) = super::run_master(ctx_clone).await {
                eprintln!("inference server error: {e}");
            }
        });
    });

    // Wait for the API server to be ready (model loading can take seconds)
    let client = Client::new();
    let mut ready = false;
    for i in 0..120 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        match client.get(format!("{server_url}/v1/models")).send().await {
            Ok(resp) if resp.status().is_success() => {
                ready = true;
                break;
            }
            _ => {
                if i > 0 && i % 10 == 0 {
                    eprintln!("still loading... ({:.0}s)", i as f64 * 0.5);
                }
            }
        }
    }

    if !ready {
        anyhow::bail!("local server did not start within 60 seconds");
    }

    // Suppress logs and take over terminal for TUI
    ::log::set_max_level(::log::LevelFilter::Off);

    // Clear any log output before entering TUI
    eprint!("\x1b[2J\x1b[H"); // clear screen + home cursor

    run_remote(&server_url).await
}
