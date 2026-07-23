from voidr_echo_runner.evaluator import TrajectoryEntry, evaluate_trajectory
from voidr_echo_runner.models import FlowAssert


def _traj(*states: str) -> list[TrajectoryEntry]:
    return [
        TrajectoryEntry(state=s, turn=i + 1, utterance=f"fala {s}", timestamp_ms=i)
        for i, s in enumerate(states)
    ]

FLOW_ASSERT = FlowAssert(
    must_visit=["saudacao", "identificacao", "diagnostico_saldo", "envio_deep_link"],
    must_not_visit=["transferencia_humano", "jornada_errada"],
    max_turns=14,
)


def test_passes_when_all_states_visited():
    trajectory = _traj("saudacao", "identificacao", "diagnostico_saldo", "envio_deep_link")
    result = evaluate_trajectory(FLOW_ASSERT, trajectory, agent_turns=5, end_reason="completed")
    assert result.status == "passed"
    assert result.error_message is None


def test_fails_on_must_not_visit_with_context():
    trajectory = _traj("saudacao", "identificacao", "jornada_errada")
    result = evaluate_trajectory(FLOW_ASSERT, trajectory, agent_turns=4, end_reason="completed")
    assert result.status == "failed"
    assert "must_not_visit violado" in result.error_message
    assert "jornada_errada" in result.error_message
    assert "turno 3" in result.error_message


def test_fails_on_missing_must_visit():
    trajectory = _traj("saudacao", "identificacao")
    result = evaluate_trajectory(FLOW_ASSERT, trajectory, agent_turns=3, end_reason="abandoned")
    assert result.status == "failed"
    assert "must_visit não atingido" in result.error_message
    assert "diagnostico_saldo" in result.error_message
    assert "envio_deep_link" in result.error_message


def test_fails_on_max_turns_exceeded():
    trajectory = _traj("saudacao", "identificacao", "diagnostico_saldo", "envio_deep_link")
    result = evaluate_trajectory(FLOW_ASSERT, trajectory, agent_turns=15, end_reason="completed")
    assert result.status == "failed"
    assert "max_turns excedido" in result.error_message


def test_fails_on_unexpected_end_reason():
    trajectory = _traj("saudacao", "identificacao", "diagnostico_saldo", "envio_deep_link")
    result = evaluate_trajectory(FLOW_ASSERT, trajectory, agent_turns=5, end_reason="abandoned")
    assert result.status == "failed"
    assert "sem estado terminal esperado" in result.error_message


def test_transport_error_is_reported():
    result = evaluate_trajectory(FLOW_ASSERT, [], agent_turns=0, end_reason=None,
                                 transport_error="connection refused")
    assert result.status == "failed"
    assert "erro de transporte" in result.error_message
