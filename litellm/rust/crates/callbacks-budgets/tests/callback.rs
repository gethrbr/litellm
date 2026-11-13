use std::sync::Arc;

use litellm_callbacks::{
    BudgetPlan, BudgetPlanCounter, BudgetReservation, Callback, Field, FieldSet, FieldValues, Hook,
    HookInput, PatchOp, Verdict,
};
use litellm_callbacks_budgets::{Budgets, Reservation, Shared};
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};
use rstest::rstest;
use serde_json::json;

fn counter(key: &str, max_budget: f64, entity_type: &str, entity_id: &str) -> BudgetPlanCounter {
    BudgetPlanCounter {
        counter_key: Some(key.to_owned()),
        max_budget: Some(max_budget),
        fallback_spend: Some(0.0),
        entity_type: Some(entity_type.to_owned()),
        entity_id: Some(entity_id.to_owned()),
        spend_log_entity_id: None,
    }
}

fn plan(counters: Vec<BudgetPlanCounter>, cost: f64) -> BudgetPlan {
    BudgetPlan {
        counters: Some(counters),
        reservation_cost: Some(cost),
        input_cost: Some(cost / 4.0),
        input_tokens: Some(12),
        throttle_counter: None,
        fail_closed: Some(false),
        status_code: Some(400),
        reservation_id: Some("reservation-1".to_owned()),
    }
}

fn store(seed: &[(&str, f64)]) -> Arc<SpendCounters> {
    let store = Arc::new(SpendCounters::new(None));
    for (key, value) in seed {
        store
            .set_local(
                &SpendCounter::parse(key).unwrap(),
                Number::Float(*value),
                None,
            )
            .unwrap();
    }
    store
}

fn value(store: &SpendCounters, key: &str) -> Option<f64> {
    store
        .get_local(&SpendCounter::parse(key).unwrap())
        .unwrap()
        .map(Number::as_f64)
}

const KEY: &str = "spend:key:hashed";

fn pre_call(plan: &BudgetPlan) -> HookInput {
    HookInput {
        hook: Hook::PreCall,
        fields: FieldValues::new(FieldSet::of(&[Field::BudgetPlan]))
            .with(Field::BudgetPlan, serde_json::to_value(plan).unwrap()),
        call: None,
        state: None,
    }
}

async fn held(callback: &Shared, plan: &BudgetPlan) -> BudgetReservation {
    let Some(Verdict::Patch(patch)) = callback.handle(&pre_call(plan)).await.unwrap() else {
        panic!("expected a reservation write");
    };
    let [PatchOp::Set { field, value }] = patch.ops.as_slice() else {
        panic!("expected one set");
    };
    assert_eq!(*field, Field::BudgetReservation);
    serde_json::from_value(value.clone()).unwrap()
}

fn settle_input(record: &BudgetReservation) -> Reservation {
    let mut wire = serde_json::to_value(record).unwrap();
    wire["finalized"] = json!(false);
    serde_json::from_value(wire).unwrap()
}

#[rstest]
#[tokio::test]
async fn the_callback_holds_at_pre_call_and_writes_the_hold_as_the_reservation() {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Arc::new(Budgets::new(Arc::clone(&store)));
    let callback = Shared(Arc::clone(&budgets));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let record = held(&callback, &plan).await;
    assert_eq!(value(&store, KEY), Some(3.0));
    assert_eq!(record.reserved_cost, Some(2.0));
    assert_eq!(record.input_cost, Some(0.5));
    assert_eq!(record.input_tokens, Some(12));
    assert_eq!(record.finalized, None);
    let settled = budgets
        .settle(settle_input(&record), 0.5, true)
        .await
        .unwrap();
    assert!(settled.reservation.finalized);
    assert!(settled.reseed.is_empty());
    assert_eq!(value(&store, KEY), Some(1.5));
    let again = budgets
        .settle(settled.reservation, 0.0, true)
        .await
        .unwrap();
    assert!(again.reservation.finalized);
    assert_eq!(value(&store, KEY), Some(1.5));
}

#[rstest]
#[tokio::test]
async fn no_plan_reserves_nothing() {
    let store = store(&[(KEY, 1.0)]);
    let callback = Shared(Arc::new(Budgets::new(Arc::clone(&store))));
    let input = HookInput {
        hook: Hook::PreCall,
        fields: FieldValues::new(FieldSet::of(&[Field::BudgetPlan])),
        call: None,
        state: None,
    };
    assert_eq!(callback.handle(&input).await.unwrap(), None);
    assert_eq!(value(&store, KEY), Some(1.0));
}

#[rstest]
#[case::no_counters(BudgetPlan { counters: None, ..plan(vec![], 2.0) })]
#[case::no_status_code(BudgetPlan { status_code: None, ..plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0) })]
#[tokio::test]
async fn a_plan_missing_a_required_member_fails_without_holding(#[case] plan: BudgetPlan) {
    let store = store(&[(KEY, 1.0)]);
    let callback = Shared(Arc::new(Budgets::new(Arc::clone(&store))));
    assert!(callback.handle(&pre_call(&plan)).await.is_err());
    assert_eq!(value(&store, KEY), Some(1.0));
}

#[rstest]
#[case::once(&[0.25])]
#[case::twice_on_the_same_cost(&[0.25, 0.25])]
#[case::grown_then_shrunk(&[0.75, 0.25])]
#[tokio::test]
async fn a_settlement_through_the_reservation_dict_lands_on_the_last_cost(#[case] costs: &[f64]) {
    let store = store(&[(KEY, 1.0)]);
    let budgets = Arc::new(Budgets::new(Arc::clone(&store)));
    let callback = Shared(Arc::clone(&budgets));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let mut wire = serde_json::to_value(settle_input(&held(&callback, &plan).await)).unwrap();
    for cost in costs {
        let reservation = serde_json::from_value(wire).unwrap();
        let settled =
            serde_json::to_value(budgets.settle(reservation, *cost, false).await.unwrap()).unwrap();
        wire = json!({
            "reserved_cost": settled["reserved_cost"],
            "entries": settled["entries"],
            "finalized": settled["finalized"],
            "input_cost": settled["input_cost"],
            "input_tokens": settled["input_tokens"],
        });
    }
    assert_eq!(value(&store, KEY), Some(1.0 + costs[costs.len() - 1]));
    assert_eq!(wire["finalized"], json!(false));
}

#[rstest]
#[tokio::test]
async fn a_refusal_aborts_with_the_typed_budget_error() {
    let store = store(&[(KEY, 10.0)]);
    let callback = Shared(Arc::new(Budgets::new(Arc::clone(&store))));
    let plan = plan(vec![counter(KEY, 10.0, "Key", "hashed")], 2.0);
    let Some(Verdict::Abort(abort)) = callback.handle(&pre_call(&plan)).await.unwrap() else {
        panic!("expected an abort");
    };
    assert_eq!(abort.status, 400);
    let error = abort.error.unwrap();
    assert_eq!(error.kind, "budget_exceeded");
    assert_eq!(
        serde_json::Value::Object(error.fields),
        json!({
            "current_cost": 12.0,
            "max_budget": 10.0,
            "message": "Budget has been exceeded! Key=hashed Current cost: 12.0, Max budget: 10.0",
            "entity_type": "key",
            "entity_id": "hashed",
        })
    );
    assert_eq!(value(&store, KEY), Some(10.0));
}
