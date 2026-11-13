use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Abort, AbortError, Activation, BudgetReservation, Callback, CallbackError, Field, FieldSet,
    Hook, HookInput, HookOutput, Manifest, OptOutBy, Patch, ReservationEntry, Selection,
    Subscription, Verdict, typed,
};
use litellm_redis_store::spend_counters::SpendCounters;
use serde::Serialize;
use serde_json::{Map, json};

use crate::{
    engine::{self, EngineError, Reservation, UNAVAILABLE},
    plan::Plan,
};

pub const NAME: &str = "budgets";

#[derive(Debug, Serialize)]
pub struct Settled {
    #[serde(flatten)]
    pub reservation: Reservation,
    pub reseed: Vec<String>,
}

pub struct Budgets {
    store: Arc<SpendCounters>,
    manifest: Manifest,
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection {
            activation: Activation::Required,
            opt_out_by: vec![OptOutBy::Admin],
            ..Selection::named(NAME)
        },
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::BudgetPlan]),
            writes: FieldSet::of(&[Field::BudgetReservation]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: false,
    }
}

fn abort(error: &EngineError, status: u16) -> Abort {
    let (status, kind, fields) = match error {
        EngineError::Exceeded(exceeded) => (
            status,
            "budget_exceeded",
            Map::from_iter([
                ("current_cost".to_owned(), json!(exceeded.current_cost)),
                ("max_budget".to_owned(), json!(exceeded.max_budget)),
                ("message".to_owned(), json!(exceeded.message)),
                ("entity_type".to_owned(), json!(exceeded.entity_type)),
                ("entity_id".to_owned(), json!(exceeded.entity_id)),
            ]),
        ),
        EngineError::Unavailable => (
            503,
            "budget_unavailable",
            Map::from_iter([("detail".to_owned(), json!(UNAVAILABLE))]),
        ),
        other => (
            401,
            "budget_reservation_failed",
            Map::from_iter([("message".to_owned(), json!(other.to_string()))]),
        ),
    };
    Abort {
        error: Some(AbortError {
            kind: kind.to_owned(),
            fields,
        }),
        ..Abort::new(status, error.to_string())
    }
}

fn reservation_record(reservation: &Reservation) -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(reservation.reserved_cost),
        entries: Some(
            reservation
                .entries
                .iter()
                .map(|entry| ReservationEntry {
                    counter_key: Some(entry.counter_key.clone()),
                    entity_type: Some(entry.entity_type.clone()),
                    entity_id: Some(entry.entity_id.clone()),
                    reserved_cost: Some(entry.reserved_cost),
                    applied_adjustment: Some(entry.applied_adjustment),
                })
                .collect(),
        ),
        finalized: None,
        input_cost: Some(reservation.input_cost),
        input_tokens: reservation.input_tokens,
    }
}

impl Budgets {
    pub fn new(store: Arc<SpendCounters>) -> Self {
        Self {
            store,
            manifest: manifest(),
        }
    }

    pub async fn settle(
        &self,
        mut reservation: Reservation,
        actual: f64,
        finalize: bool,
    ) -> Result<Settled, EngineError> {
        let reseed = engine::settle(&self.store, &mut reservation, actual, finalize).await?;
        Ok(Settled {
            reservation,
            reseed,
        })
    }

    async fn admit(&self, input: &HookInput) -> HookOutput {
        let Some(record) = input.fields.get::<typed::BudgetPlan>()? else {
            return Ok(None);
        };
        let plan =
            Plan::try_from(record).map_err(|error| CallbackError::Failed(error.to_string()))?;
        match engine::reserve(&self.store, &plan).await {
            Ok(None) => Ok(None),
            Ok(Some(reservation)) => {
                let record = serde_json::to_value(reservation_record(&reservation))
                    .map_err(|error| CallbackError::Failed(error.to_string()))?;
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::BudgetReservation, record),
                )))
            }
            Err(error) => Ok(Some(Verdict::Abort(abort(&error, plan.status_code)))),
        }
    }
}

pub struct Shared(pub Arc<Budgets>);

impl Callback for Shared {
    fn manifest(&self) -> &Manifest {
        &self.0.manifest
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::PreCall => self.0.admit(input).await,
                _ => Ok(None),
            }
        })
    }
}
