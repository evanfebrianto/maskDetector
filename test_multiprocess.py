from multiprocessing import Process


def print_func(continent='Asia'):
    global isSendingEmail
    if isSendingEmail:
        print('Entering print_func')
    isSendingEmail = False

if __name__ == "__main__":  # confirms that the code is under main function
    isSendingEmail = False
    counter = 0
    procs = []
    while counter < 5:
        isSendingEmail = True
        print('While....')
        proc = Process(target=print_func)  # instantiating without any argument
        procs.append(proc)
        proc.start()
        counter += 1

    # complete the processes
    for proc in procs:
        proc.join()
