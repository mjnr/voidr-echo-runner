import asyncio
from types import MethodType

from voidr_echo_runner.runner import CallResult, CallRunner


class CleanupTransport:
    def __init__(self):
        self.events: list[str] = []

    async def connect(self) -> None:
        self.events.append("connect")

    async def finish_audio(self, sample_rate: int) -> None:
        self.events.append("finish_audio")
        raise RuntimeError("cleanup finish failed")

    async def hangup(self) -> None:
        self.events.append("hangup")

    async def close(self) -> None:
        self.events.append("close")


def test_runner_preserves_original_error_and_runs_all_cleanup():
    async def scenario():
        transport = CleanupTransport()
        runner = CallRunner.__new__(CallRunner)
        runner.transport = transport
        runner.result = CallResult()
        runner.live = None

        async def fail_loop(self):
            raise ValueError("original streaming failure")

        runner._loop = MethodType(fail_loop, runner)
        result = await runner.run()

        assert result.transport_error == "ValueError: original streaming failure"
        assert transport.events == ["connect", "finish_audio", "hangup", "close"]

    asyncio.run(scenario())


def test_runner_cleanup_is_safe_to_repeat():
    async def scenario():
        transport = CleanupTransport()
        runner = CallRunner.__new__(CallRunner)
        runner.transport = transport
        runner.result = CallResult()
        runner.live = None

        async def end_loop(self):
            return None

        runner._loop = MethodType(end_loop, runner)
        await runner.run()
        runner.result = CallResult()
        await runner.run()

        assert transport.events.count("hangup") == 2
        assert transport.events.count("close") == 2

    asyncio.run(scenario())
