import threading
import time
import sys

def background():
        while True:
            time.sleep(5)
            print('nd me by typing v')


def other_function():
    print('You disarmed me! Dying now.')

# now threading1 runs regardless of user input
threading1 = threading.Thread(target=background)
threading1.daemon = True
threading1.start()

while True:
    if input() == 'v':
        other_function()
        sys.exit()
    else:
        print('not disarmed')