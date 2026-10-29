//! 用本地 HTTP 服务核验公共入口真正发出的参数，不读取用户密钥或消耗 API 用量。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use qingjian_core::{PredictionKind, PredictionRequest, Predictor};
use qingjian_predict::{CloudPredictor, ConnectionTest, PredictConfig};
use serde_json::{Value, json};

#[test]
fn sol_connection_uses_compatible_parameters() {
    for effort in [
        "", "none", "minimal", "low", "medium", "high", "xhigh", "max",
    ] {
        let body = capture_request("gpt-6.1-sol", effort, false);
        assert_sol_parameters(&body, 4128);
        let expected = match effort {
            "" | "none" | "minimal" => "low",
            value => value,
        };
        assert_eq!(body["reasoning_effort"], expected, "effort={effort:?}");
    }
}

#[test]
fn sol_prediction_uses_the_same_compatible_parameters() {
    let body = capture_request("gpt-6.1-sol", "none", true);
    assert_sol_parameters(&body, 4296);
    assert_eq!(body["reasoning_effort"], "low");
    let user: Value =
        serde_json::from_str(body["messages"][1]["content"].as_str().unwrap()).unwrap();
    assert_eq!(user["letters"], "nihao");
}

#[test]
fn legacy_models_keep_their_parameters() {
    for (model, effort, expected) in [
        ("deepseek-flash", "none", Some("none")),
        ("deepseek-flash", "max", None),
        ("gpt-4.1-mini", "", None),
    ] {
        let body = capture_request(model, effort, false);
        assert_eq!(body["model"], model);
        assert_eq!(body["max_tokens"], 32);
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["temperature"], 0.3);
        if let Some(expected) = expected {
            assert_eq!(body["reasoning_effort"], expected);
        } else {
            assert!(body.get("reasoning_effort").is_none());
        }
    }
}

fn assert_sol_parameters(body: &Value, completion_tokens: u32) {
    assert_eq!(body["model"], "gpt-6.1-sol");
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("temperature").is_none());
    assert_eq!(body["max_completion_tokens"], completion_tokens);
    assert_eq!(body["response_format"]["type"], "json_object");
}

fn capture_request(model: &str, effort: &str, prediction: bool) -> Value {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let config = PredictConfig {
        base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
        model: model.to_owned(),
        api_key: Some("test-only-key".to_owned()),
        reasoning_effort: effort.to_owned(),
        timeout_ms: 2000,
        debounce_ms: 0,
        min_interval_ms: 0,
        ..PredictConfig::default()
    };
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        let stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        panic!("mock server received no request");
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(error) => panic!("accept: {error}"),
            }
        };
        stream.set_nonblocking(false).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut reader = BufReader::new(stream);
        let mut first_line = String::new();
        reader.read_line(&mut first_line).unwrap();
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse::<usize>().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        let response = json!({
            "id": "compat-test", "object": "chat.completion", "created": 0,
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{\"ok\":true}"},
                "finish_reason": "stop"
            }]
        })
        .to_string();
        write!(reader.get_mut(),
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response.len(), response
        ).unwrap();
        (first_line, serde_json::from_slice::<Value>(&body).unwrap())
    });
    let completed = if prediction {
        let mut predictor = CloudPredictor::new(&config).unwrap();
        predictor.submit(PredictionRequest {
            sequence: 7,
            manual: true,
            kind: PredictionKind::Compose,
            before: String::new(),
            after: String::new(),
            pinyin: "ni'hao".to_owned(),
            letters: "nihao".to_owned(),
            syllables: 2,
            candidates: Vec::new(),
            guess: String::new(),
            max_items: 2,
            want_sentence: false,
            text: String::new(),
            target_language: String::new(),
        });
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(reply) = predictor.poll() {
                break reply.sequence == 7;
            }
            if Instant::now() >= deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    } else {
        ConnectionTest::start(&config)
            .unwrap()
            .wait()
            .is_ok_and(|report| report.reply == "{\"ok\":true}")
    };
    let (first_line, body) = server.join().unwrap();
    assert!(completed, "public API did not receive the mock reply");
    assert_eq!(first_line, "POST /v1/chat/completions HTTP/1.1\r\n");
    body
}
