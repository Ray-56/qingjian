//! 仅按需验证真实 GPT-6.1 Sol；使用固定合成输入，默认忽略，不读取本机配置。

use std::time::{Duration, Instant};

use qingjian_core::{Prediction, PredictionKind, PredictionRequest, Predictor};
use qingjian_predict::{CloudPredictor, ConnectionTest, PredictConfig};
use serde_json::Value;

#[test]
#[ignore = "requires QINGJIAN_API_KEY and makes a paid API request"]
fn gpt61_live_connection() {
    let config = live_config();
    let report = ConnectionTest::start(&config)
        .and_then(ConnectionTest::wait)
        .unwrap_or_else(|_| panic!("gpt-6.1-sol live connection failed"));
    assert_eq!(report.model, "gpt-6.1-sol");
    let reply: Value = serde_json::from_str(&report.reply).expect("reply must be visible JSON");
    assert_eq!(reply["ok"], true);
    println!(
        "model={} kind=connection elapsed_ms={}",
        report.model,
        report.elapsed.as_millis()
    );
}

#[test]
#[ignore = "requires QINGJIAN_API_KEY and makes a paid API request"]
fn gpt61_live_composition() {
    let (reply, elapsed) = predict(PredictionKind::Compose);
    assert!(
        !reply.words.is_empty()
            || reply
                .sentence
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty()),
        "composition must produce a word or sentence"
    );
    println!(
        "model=gpt-6.1-sol kind=compose elapsed_ms={}",
        elapsed.as_millis()
    );
}

#[test]
#[ignore = "requires QINGJIAN_API_KEY and makes a paid API request"]
fn gpt61_live_translation() {
    let (reply, elapsed) = predict(PredictionKind::Translate);
    let sentence = reply
        .sentence
        .as_deref()
        .expect("translation must produce a sentence");
    assert!(!sentence.trim().is_empty());
    assert!(
        sentence.chars().any(|c| c.is_ascii_alphabetic()),
        "translation must contain English"
    );
    println!(
        "model=gpt-6.1-sol kind=translate elapsed_ms={}",
        elapsed.as_millis()
    );
}

fn live_config() -> PredictConfig {
    PredictConfig {
        base_url: "https://api.openai.com/v1".to_owned(),
        model: "gpt-6.1-sol".to_owned(),
        api_key: Some(std::env::var("QINGJIAN_API_KEY").expect("QINGJIAN_API_KEY is required")),
        reasoning_effort: String::new(),
        timeout_ms: 30_000,
        debounce_ms: 0,
        min_interval_ms: 0,
        ..PredictConfig::default()
    }
}

fn predict(kind: PredictionKind) -> (Prediction, Duration) {
    let mut predictor = CloudPredictor::new(&live_config())
        .unwrap_or_else(|_| panic!("could not create live predictor"));
    let started = Instant::now();
    predictor.submit(PredictionRequest {
        sequence: 7,
        manual: true,
        kind,
        before: String::new(),
        after: String::new(),
        pinyin: "ni'hao".to_owned(),
        letters: "nihao".to_owned(),
        syllables: 2,
        candidates: Vec::new(),
        guess: "你好".to_owned(),
        max_items: 2,
        want_sentence: true,
        text: "你好".to_owned(),
        target_language: "en".to_owned(),
    });
    let deadline = started + Duration::from_secs(30);
    loop {
        if let Some(reply) = predictor.poll() {
            assert_eq!(reply.sequence, 7);
            return (reply, started.elapsed());
        }
        assert!(
            Instant::now() < deadline,
            "live prediction exceeded 30 seconds"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}
