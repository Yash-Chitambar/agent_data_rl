from deploy.recovery import supported_shutdown


def test_shutdown_requires_explicit_support_confirmation():
    events = []
    class Backend:
        def hold(self): events.append('hold')
        def close(self): events.append('close')
    answers = iter(['', 'no', 'SUPPORTED'])
    def answer(prompt):
        assert events == ['hold']
        return next(answers)
    supported_shutdown(Backend(), answer)
    assert events == ['hold', 'close']
