/// Memory-leak detection test for long-running proxy workloads.
///
/// Simulates real-world browsing-like traffic through the proxy's SOCKS5 port:
/// domain-based CONNECT to external HTTP(S) hosts, short-lived connections with
/// small payloads, concurrent bursts, repeated over many rounds.  Samples RSS
/// after each round and fits a linear regression to detect unbounded growth.
///
/// Intended for use with a real proxy config (e.g. ECH-TLS-tunnel) against a
/// live server — requires network access.
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::bench_memory::measure_rss;
use crate::socks5_client::socks5_connect_domain;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemleakResult {
    pub rounds: usize,
    pub connections_per_round: usize,
    pub concurrency: usize,
    pub rss_samples_mb: Vec<f64>,
    pub slope_kb_per_round: f64,
    pub r_squared: f64,
    pub verdict: String,
}

struct Target {
    host: &'static str,
    port: u16,
    request: &'static str,
}

const TARGETS: &[Target] = &[
    Target {
        host: "www.gstatic.com",
        port: 80,
        request: "GET /generate_204 HTTP/1.1\r\nHost: www.gstatic.com\r\nConnection: close\r\n\r\n",
    },
    Target {
        host: "cp.cloudflare.com",
        port: 80,
        request: "GET / HTTP/1.1\r\nHost: cp.cloudflare.com\r\nConnection: close\r\n\r\n",
    },
    Target {
        host: "detectportal.firefox.com",
        port: 80,
        request: "GET /success.txt HTTP/1.1\r\nHost: detectportal.firefox.com\r\nConnection: close\r\n\r\n",
    },
];

async fn do_one_request(proxy: SocketAddr, target: &Target) -> bool {
    let Ok(mut stream) = socks5_connect_domain(proxy, target.host, target.port).await else {
        return false;
    };
    if stream.write_all(target.request.as_bytes()).await.is_err() {
        return false;
    }
    let mut buf = vec![0u8; 4096];
    // Read at least the status line; don't care about full body.
    match tokio::time::timeout(Duration::from_secs(10), stream.read(&mut buf)).await {
        Ok(Ok(n)) if n > 0 => {}
        _ => return false,
    }
    let _ = stream.shutdown().await;
    true
}

/// Run one round: `conns_per_round` connections spread across `concurrency` workers.
async fn run_round(proxy: SocketAddr, conns_per_round: usize, concurrency: usize) -> usize {
    let per_worker = conns_per_round / concurrency.max(1);
    let remainder = conns_per_round % concurrency.max(1);

    let mut handles = Vec::with_capacity(concurrency);
    for w in 0..concurrency {
        let count = per_worker + if w < remainder { 1 } else { 0 };
        let offset = w;
        handles.push(tokio::spawn(async move {
            let mut ok = 0usize;
            for i in 0..count {
                let idx = offset + i * concurrency;
                let target = &TARGETS[idx % TARGETS.len()];
                if do_one_request(proxy, target).await {
                    ok += 1;
                }
            }
            ok
        }));
    }

    let mut total_ok = 0;
    for h in handles {
        if let Ok(n) = h.await {
            total_ok += n;
        }
    }
    total_ok
}

fn linear_regression(ys: &[f64]) -> (f64, f64, f64) {
    let n = ys.len() as f64;
    let xs: Vec<f64> = (0..ys.len()).map(|i| i as f64).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return (0.0, sum_y / n, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let mean_y = sum_y / n;
    let ss_tot: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let predicted = slope * x + intercept;
            (y - predicted).powi(2)
        })
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    (slope, intercept, r_squared)
}

/// Maximum tolerable RSS growth rate (KB per round). Accounts for normal
/// allocator fragmentation and OS page-rounding.  A true leak in the
/// relay / transport / TLS path will far exceed this.
const MAX_SLOPE_KB_PER_ROUND: f64 = 50.0;

pub async fn bench_memleak(
    proxy: SocketAddr,
    proxy_pid: u32,
    rounds: usize,
    conns_per_round: usize,
    concurrency: usize,
) -> anyhow::Result<MemleakResult> {
    eprintln!(
        "  memleak: {rounds} rounds × {conns_per_round} connections, concurrency={concurrency}"
    );

    // Warmup — prime TLS session caches, DNS caches, etc.
    eprintln!("  memleak: warming up (50 connections)...");
    run_round(proxy, 50.min(conns_per_round), concurrency).await;
    tokio::time::sleep(Duration::from_secs(2)).await;

    let mut rss_samples = Vec::with_capacity(rounds);

    for round in 0..rounds {
        let start = Instant::now();
        let ok = run_round(proxy, conns_per_round, concurrency).await;
        let elapsed = start.elapsed();

        // Let the proxy settle (GC-like allocator coalescing, tokio task cleanup).
        tokio::time::sleep(Duration::from_secs(2)).await;

        let rss = measure_rss(proxy_pid)?;
        let rss_mb = rss as f64 / 1_048_576.0;
        rss_samples.push(rss_mb);

        eprintln!(
            "  memleak: round {}/{rounds}  ok={ok}/{conns_per_round}  {:.1}s  RSS={:.1} MB",
            round + 1,
            elapsed.as_secs_f64(),
            rss_mb,
        );
    }

    let (slope_mb, _intercept, r_squared) = linear_regression(&rss_samples);
    let slope_kb = slope_mb * 1024.0;

    let verdict = if slope_kb > MAX_SLOPE_KB_PER_ROUND && r_squared > 0.7 {
        format!(
            "LEAK SUSPECTED: RSS growing {slope_kb:.1} KB/round (R²={r_squared:.2}), threshold={MAX_SLOPE_KB_PER_ROUND} KB/round"
        )
    } else {
        format!("OK: RSS slope {slope_kb:.1} KB/round (R²={r_squared:.2})")
    };

    eprintln!("  memleak: {verdict}");

    Ok(MemleakResult {
        rounds,
        connections_per_round: conns_per_round,
        concurrency,
        rss_samples_mb: rss_samples,
        slope_kb_per_round: slope_kb,
        r_squared,
        verdict,
    })
}
