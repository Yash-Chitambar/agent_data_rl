"""Keep power until an operator explicitly confirms physical support for shutdown."""
import time


def supported_shutdown(backend, read_line=input):
    backend.hold()
    print('Holding powered. Secure the arm and any held object. Type SUPPORTED to disable torque.', flush=True)
    while True:
        try:
            answer = read_line('support confirmation > ')
        except (EOFError, KeyboardInterrupt):
            print('No support confirmation; maintaining power.', flush=True)
            time.sleep(.5)
            continue
        if answer.strip() == 'SUPPORTED':
            backend.close()
            return
        print('Still holding; torque remains enabled.', flush=True)
