use cake_core::cake::auth;

#[divan::bench]
fn compute_hmac_32b(bencher: divan::Bencher) {
    let key = b"my-secret-cluster-key";
    let data = [0xABu8; 32];
    bencher.bench_local(|| auth::compute_hmac(key, &data));
}

#[divan::bench]
fn compute_hmac_1kb(bencher: divan::Bencher) {
    let key = b"my-secret-cluster-key";
    let data = vec![0xCDu8; 1024];
    bencher
        .counter(divan::counter::BytesCount::new(1024usize))
        .bench_local(|| auth::compute_hmac(key, &data));
}

#[divan::bench]
fn constant_time_eq_match(bencher: divan::Bencher) {
    let a = [0x42u8; 32];
    let b = [0x42u8; 32];
    bencher.bench_local(|| auth::constant_time_eq(&a, &b));
}

#[divan::bench]
fn full_auth_handshake(bencher: divan::Bencher) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let key = "bench-secret-key";
    bencher.bench_local(|| {
        rt.block_on(async {
            let (mut master_side, mut worker_side) = tokio::io::duplex(4096);
            let (r1, r2) = tokio::join!(
                auth::authenticate_as_master(&mut master_side, key),
                auth::authenticate_as_worker(&mut worker_side, key),
            );
            r1.unwrap();
            r2.unwrap();
        })
    });
}
