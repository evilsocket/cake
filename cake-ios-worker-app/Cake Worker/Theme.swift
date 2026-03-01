//
//  Theme.swift
//  Cake
//
//  Color palette and styling matching the cake web UI.
//

import SwiftUI

// MARK: - Color Palette

extension Color {
    // Brand colors (purple spectrum, hue ~270°)
    static let brand50  = Color(red: 0.96, green: 0.95, blue: 1.00)
    static let brand100 = Color(red: 0.91, green: 0.88, blue: 1.00)
    static let brand200 = Color(red: 0.85, green: 0.80, blue: 1.00)
    static let brand300 = Color(red: 0.75, green: 0.67, blue: 1.00)
    static let brand400 = Color(red: 0.64, green: 0.54, blue: 1.00)
    static let brand500 = Color(red: 0.55, green: 0.44, blue: 1.00)
    static let brand600 = Color(red: 0.46, green: 0.33, blue: 0.91)
    static let brand700 = Color(red: 0.39, green: 0.29, blue: 0.82)
    static let brand800 = Color(red: 0.33, green: 0.25, blue: 0.73)
    static let brand900 = Color(red: 0.26, green: 0.17, blue: 0.60)

    // Surface colors (dark theme with slight purple tint)
    static let surface0   = Color(red: 0.10, green: 0.11, blue: 0.15)
    static let surface50  = Color(red: 0.14, green: 0.15, blue: 0.20)
    static let surface100 = Color(red: 0.18, green: 0.18, blue: 0.24)
    static let surface200 = Color(red: 0.21, green: 0.22, blue: 0.31)
    static let surface300 = Color(red: 0.28, green: 0.29, blue: 0.39)
    static let surface400 = Color(red: 0.38, green: 0.42, blue: 0.55)
    static let surface500 = Color(red: 0.52, green: 0.56, blue: 0.66)
    static let surface600 = Color(red: 0.67, green: 0.67, blue: 0.77)
    static let surface700 = Color(red: 0.82, green: 0.82, blue: 0.87)
    static let surface800 = Color(red: 0.91, green: 0.91, blue: 0.93)

    // Accent colors (teal/cyan)
    static let accent400 = Color(red: 0.18, green: 0.83, blue: 0.75)
    static let accent500 = Color(red: 0.08, green: 0.72, blue: 0.65)
    static let accent600 = Color(red: 0.05, green: 0.58, blue: 0.53)

    // Status
    static let success = Color(red: 0.13, green: 0.77, blue: 0.37)
    static let warning = Color(red: 0.98, green: 0.75, blue: 0.14)
    static let danger  = Color(red: 0.94, green: 0.27, blue: 0.27)
}

// MARK: - View Modifiers

struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Color.surface50)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.surface200, lineWidth: 1)
            )
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    var isEnabled: Bool = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 16, weight: .semibold))
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity)
            .background(
                isEnabled
                    ? (configuration.isPressed ? Color.brand600 : Color.brand500)
                    : Color.surface300
            )
            .cornerRadius(12)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .medium))
            .foregroundColor(Color.brand400)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(configuration.isPressed ? Color.surface100 : Color.surface50)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.surface200, lineWidth: 1)
            )
    }
}

extension View {
    func cardStyle() -> some View {
        modifier(CardStyle())
    }
}
