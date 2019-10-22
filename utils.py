# Learn model t only
# Creating a dataset for learning different T values

def stateAction2forwardDyn(states, actions):
    data_in = np.concatenate((states[:-1, :], actions[:-1, :]), axis=1)
    data_out = states[1:, :]
    return [data_in, data_out]

def create_dataset(data, cfg):
    raise NotImplementedError("TODO merge these create dataset functions into a modular single function")

def create_dataset_t_only(states):
    """
    Creates a dataset with an entry for how many timesteps in the future
    corresponding entries in the labels are
    :param states: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for i in range(states.shape[0]):  # From one state p
        for j in range(i + 1, states.shape[0]):
            # This creates an entry for a given state concatenated with a number t of time steps
            data_in.append(np.hstack((states[i], j - i)))
            # This creates an entry for the state t timesteps in the future
            data_out.append(states[j])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out

def create_dataset_t_pid(states, P, I, D, goal):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future
    :param states: A 2d np array. Each row is a state
    """
    data_in, data_out = [], []
    for i in range(states.shape[0]):  # From one state p
        for j in range(i + 1, states.shape[0]):
            # This creates an entry for a given state concatenated
            # with a number t of time steps as well as the PID parameters
            data_in.append(np.hstack((states[i], j - i, P, I, D, goal))) # Did we want this to just have the goal or all parameters?
            data_out.append(states[j])
    return data_in, data_out



def create_dataset_no_t(data):
    """
    Creates a dataset for learning how one state progresses to the next
    """
    data_in = []
    data_out = []
    for sequence in data:
        for i in range(sequence.states.shape[0] - 1):
            data_in.append(np.hstack((sequence.states[i], sequence.actions[i])))
            data_out.append(sequence.states[i + 1])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset(data):
    dataset_in = None
    dataset_out = None
    for sequence in data:
        inputs, outputs = create_dataset_t_only(sequence.states)
        if dataset_in is None:
            dataset_in = inputs
            dataset_out = outputs
        else:
            dataset_in = np.concatenate((dataset_in, inputs), axis=0)
            dataset_out = np.concatenate((dataset_out, outputs), axis=0)
    return [dataset_in, dataset_out]