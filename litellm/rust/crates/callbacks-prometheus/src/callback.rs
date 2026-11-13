use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest,
    Selection, Subscription,
};
use serde::{Deserialize, Serialize};

use crate::{
    LogLine, Step,
    ops::{ErrorKind, OpError},
    record,
    store::Store,
    success::AttemptFacts,
};

pub trait LogSink: Send + Sync {
    fn log(&self, line: &LogLine);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Success,
    Failure,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Attempt {
    pub event: Event,
    pub facts: AttemptFacts,
    #[serde(default)]
    pub input_sequence_length: Option<String>,
    #[serde(default)]
    pub input_sequence_length_failed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopKind {
    Value,
    Key,
    Type,
    Runtime,
    Carried,
}

impl From<ErrorKind> for StopKind {
    fn from(kind: ErrorKind) -> Self {
        match kind {
            ErrorKind::Value => Self::Value,
            ErrorKind::Key => Self::Key,
            ErrorKind::Type => Self::Type,
            ErrorKind::Runtime => Self::Runtime,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stop {
    pub step: Step,
    pub kind: StopKind,
    pub message: String,
}

impl Stop {
    fn raised(step: Step, error: CallbackError) -> Self {
        let error = OpError::from(error);
        Self {
            step,
            kind: error.kind.into(),
            message: error.message,
        }
    }

    const fn carried(step: Step) -> Self {
        Self {
            step,
            kind: StopKind::Carried,
            message: String::new(),
        }
    }
}

const SUCCESS_STEPS: [Step; 4] = [
    Step::SuccessHead,
    Step::SuccessLimits,
    Step::SuccessLatency,
    Step::SuccessTail,
];
const FAILURE_STEPS: [Step; 1] = [Step::Failure];

pub struct Prometheus {
    store: Arc<Store>,
    sink: Arc<dyn LogSink>,
    manifest: Manifest,
}

impl Prometheus {
    pub fn new(store: Arc<Store>, sink: Arc<dyn LogSink>) -> Self {
        Self {
            store,
            sink,
            manifest: Self::declared(),
        }
    }

    pub fn declared() -> Manifest {
        Manifest {
            selection: Selection {
                name: "prometheus".to_owned(),
                aliases: Vec::new(),
                activation: Activation::Required,
                opt_out_by: Vec::new(),
                honors_no_log: false,
            },
            subscriptions: vec![Subscription {
                reads: FieldSet::of(&[Field::DeprecatedKwargs]),
                awaited: true,
                ..Subscription::on(Hook::AttemptEnded)
            }],
            has_request_state: false,
        }
    }

    pub fn run(&self, attempt: &Attempt) -> Result<(), Stop> {
        let steps: &[Step] = match attempt.event {
            Event::Success => &SUCCESS_STEPS,
            Event::Failure => &FAILURE_STEPS,
        };
        for &step in steps {
            if step == Step::SuccessLatency && attempt.input_sequence_length_failed {
                return Err(Stop::carried(step));
            }
            let input_sequence_length = (step == Step::SuccessLatency)
                .then(|| attempt.input_sequence_length.clone())
                .flatten();
            let recorded = record(&self.store, &attempt.facts, step, input_sequence_length);
            for line in &recorded.logs {
                self.sink.log(line);
            }
            recorded
                .outcome
                .map_err(|error| Stop::raised(step, error))?;
        }
        Ok(())
    }

    fn attempt_ended(&self, input: &HookInput) -> HookOutput {
        let Some(raw) = input.fields.raw(Field::DeprecatedKwargs)? else {
            return Ok(None);
        };
        let attempt: Attempt = serde_json::from_value(raw.clone())
            .map_err(|error| CallbackError::Failed(format!("prometheus attempt: {error}")))?;
        self.run(&attempt).map_err(|stop| {
            CallbackError::Failed(serde_json::to_string(&stop).unwrap_or(stop.message))
        })?;
        Ok(None)
    }
}

impl Callback for Prometheus {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::AttemptEnded => self.attempt_ended(input),
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::FutureExt;
    use litellm_callbacks::FieldValues;
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::{FamilySpec, Kind, ManualClock, SeriesLimits};

    const FAMILIES: [(&str, Kind); 7] = [
        ("litellm_requests_metric", Kind::Counter),
        ("litellm_spend_metric", Kind::Counter),
        ("litellm_total_tokens_metric", Kind::Counter),
        ("litellm_input_tokens_metric", Kind::Counter),
        ("litellm_output_tokens_metric", Kind::Counter),
        ("litellm_llm_api_failed_requests_metric", Kind::Counter),
        ("litellm_proxy_total_requests_metric", Kind::Counter),
    ];

    #[derive(Default)]
    struct Lines(Mutex<Vec<LogLine>>);

    impl LogSink for Lines {
        fn log(&self, line: &LogLine) {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(line.clone());
        }
    }

    #[fixture]
    fn prometheus() -> Prometheus {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        for (name, kind) in FAMILIES {
            store
                .register(FamilySpec {
                    declared: name.to_owned(),
                    documentation: "doc".to_owned(),
                    kind,
                    labelnames: vec!["team".to_owned()],
                    excluded: Vec::new(),
                    buckets: Vec::new(),
                    most_recent: false,
                    wrapped: false,
                    shares_overflow: true,
                })
                .unwrap_or_else(|error| panic!("{error}"));
        }
        Prometheus::new(Arc::new(store), Arc::new(Lines::default()))
    }

    fn attempt(event: &str, completion_tokens: Value, failed: bool) -> Value {
        json!({
            "event": event,
            "input_sequence_length_failed": failed,
            "facts": {
                "payload": {
                    "metadata": {
                        "user_api_key_user_id": "u",
                        "user_api_key_hash": "hashed",
                        "user_api_key_alias": null,
                        "user_api_key_team_id": "team-1",
                        "user_api_key_team_alias": null,
                        "user_api_key_user_email": null,
                    },
                    "completion_tokens": completion_tokens,
                    "total_tokens": 30,
                    "prompt_tokens": 20,
                    "response_cost": 0.5,
                    "request_tags": [],
                    "model_group": "group",
                    "model_id": "id",
                    "api_base": "base",
                    "custom_llm_provider": "openai",
                },
                "settings": {},
            },
        })
    }

    fn handled(prometheus: &Prometheus, hook: Hook, value: Option<Value>) -> HookOutput {
        let mut fields = FieldValues::new(FieldSet::of(&[Field::DeprecatedKwargs]));
        if let Some(value) = value {
            fields.insert(Field::DeprecatedKwargs, value);
        }
        let input = HookInput {
            hook,
            fields,
            call: None,
            state: None,
        };
        prometheus
            .handle(&input)
            .now_or_never()
            .unwrap_or_else(|| panic!("the handler completes without waiting"))
    }

    fn total(prometheus: &Prometheus, name: &str) -> Vec<f64> {
        let id = prometheus
            .store
            .family(name)
            .unwrap_or_else(|| panic!("{name} is registered"));
        prometheus
            .store
            .snapshot(id)
            .series
            .into_iter()
            .map(|(_, series)| series.value)
            .collect()
    }

    fn stop(output: HookOutput) -> Option<Stop> {
        match output {
            Ok(_) => None,
            Err(CallbackError::Failed(text)) => {
                Some(serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}")))
            }
            Err(other) => panic!("unexpected error {other}"),
        }
    }

    #[rstest]
    fn the_manifest_is_one_awaited_attempt_ended_read_that_validates() {
        let manifest = Prometheus::declared();
        assert_eq!(manifest.validate(), Ok(()));
        assert_eq!(manifest.selection.activation, Activation::Required);
        assert_eq!(
            manifest
                .subscriptions
                .iter()
                .map(|subscription| (
                    subscription.hook,
                    subscription.awaited,
                    subscription.reads.contains(Field::DeprecatedKwargs)
                ))
                .collect::<Vec<_>>(),
            vec![(Hook::AttemptEnded, true, true)]
        );
    }

    #[rstest]
    fn a_success_runs_every_step(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            Some(attempt("success", json!(10), false)),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(total(&prometheus, "litellm_requests_metric"), vec![1.0]);
        assert_eq!(
            total(&prometheus, "litellm_output_tokens_metric"),
            vec![10.0]
        );
        assert_eq!(
            total(&prometheus, "litellm_proxy_total_requests_metric"),
            vec![1.0]
        );
    }

    #[rstest]
    fn a_failure_runs_only_the_failure_step(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            Some(attempt("failure", json!(10), false)),
        );
        assert!(matches!(output, Ok(None)));
        assert_eq!(
            total(&prometheus, "litellm_llm_api_failed_requests_metric"),
            vec![1.0]
        );
        assert!(total(&prometheus, "litellm_requests_metric").is_empty());
    }

    #[rstest]
    fn a_raise_reports_its_step_and_python_class(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            Some(attempt("success", json!("ten"), false)),
        );
        assert_eq!(
            stop(output),
            Some(Stop {
                step: Step::SuccessHead,
                kind: StopKind::Value,
                message: "could not convert string to float: 'ten'".to_owned(),
            })
        );
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    fn a_carried_python_error_stops_before_latency(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            Some(attempt("success", json!(10), true)),
        );
        assert_eq!(
            stop(output),
            Some(Stop {
                step: Step::SuccessLatency,
                kind: StopKind::Carried,
                message: String::new(),
            })
        );
        assert_eq!(total(&prometheus, "litellm_requests_metric"), vec![1.0]);
        assert!(total(&prometheus, "litellm_proxy_total_requests_metric").is_empty());
    }

    #[rstest]
    #[case::other_hook(Hook::RouterEvent, Some(json!({"event": "success"})))]
    #[case::no_value(Hook::AttemptEnded, None)]
    fn nothing_else_records(
        prometheus: Prometheus,
        #[case] hook: Hook,
        #[case] value: Option<Value>,
    ) {
        assert!(matches!(handled(&prometheus, hook, value), Ok(None)));
        assert!(
            FAMILIES
                .iter()
                .all(|(name, _)| total(&prometheus, name).is_empty())
        );
    }

    #[rstest]
    fn a_malformed_attempt_is_an_error(prometheus: Prometheus) {
        let output = handled(
            &prometheus,
            Hook::AttemptEnded,
            Some(json!({"event": "nope"})),
        );
        assert!(
            matches!(output, Err(CallbackError::Failed(text)) if text.starts_with("prometheus attempt: "))
        );
    }
}
