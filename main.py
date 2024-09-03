import sys
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)



#############################
# ERROR MESSAGES DEFINITION #
#############################
ERROR_MESSAGE_SYS_EXIT_INVALID_ARGUMENT = "Error: FAILED argument provided is invalid: "
ERROR_MESSAGE_SYS_EXIT_UNEXPECTED_BEHAVIOUR = "Error: FAILED could not proceed with main.py call"
ERROR_MESSAGE_SYS_EXIT_NO_MATCH = "Error: FAILED could not find an environment to proceed with function call"



############################################
# GLOBAL VARIABLE AND PARAMETER DEFINITION #
############################################
SUCCESS = True
FAILED = False
VALID_ARGUMENT_LIST_FOR_SYS = ["main.py", "rdm", "-p", "geo", "stat", "normal", "brown"]

VALID_ARGUMENT_LIST_FOR_GEOPAIRING_DISTRIBUTION_NAME = ["normal", "brown"]
_GEOPAIRING_RANDOM_VECTOR_LEN = int(1e6)
_RANDOM_WALK_NON_ZERO_DRIFT_VALUE = 0.01


########################################
# LOCAL AND GLOBAL FUNCTION DEFINITION #
########################################
def mean(s):
    out_s = 0
    for elem in s:
        out_s = out_s + elem
    return (out_s/len(s))



def var(s):
    u = mean(s)
    out_s = 0
    for elem in s:
        out_s = out_s + ((u-elem)**2)
    return ((1/len(s))*out_s)



def dev(s):
    return (var(s))**0.5
    
    

def list_sum_mult(s1, s2):
    out = 0
    if len(s1)!= len(s2):
        return None
    for i in range(len(s1)):
        out = out + (s1[i]*s2[i])
    return out



def list_sub(s, x):
    return [(item-x) for item in s]


    
def autocorrelation_1(s):
    """
    @brief: autocorrelation computation by shifting signal
    @date: 2024 septembre 02
    """
    out = []
    
    if isinstance(s, np.ndarray):
        s = s.tolist()
    temp_s = s.copy()
    
    N = len(s)
    for i in range(N):
        out.append((1/N)*list_sum_mult(s, temp_s))
        temp_s.pop(0)
        temp_s.append(0)
    return out



def autocorrelation_2(s):
    """
    @brief: autocorrelation computation by shifting signal and norming the product result in order to have -1 <= R(k) <= 1
    @comment: not appropriate for Inertial Measurement Unit (IMU) angle random walk extraction 
    @date: 2024 septembre 03
    """
    out = []
    sig = dev(s)
    u = mean(s)
    
    if isinstance(s, np.ndarray):
        s = s.tolist()
    temp_s = s.copy()
    
    N = len(s)
    for i in range(N):
        out.append((1/((N)*(sig**2)))*list_sum_mult(list_sub(s,u), list_sub(temp_s,u)))
        temp_s.pop(0)
        temp_s.append(0)
    return out
        


def generate_random_vector(vector_size, distribution=None):
    """
    @brief: generate random vector with specific noise according to the distribution argument provided to the function
    @date: 2024 August 30
    """
    temp_list = []
    if (None != distribution):
        match distribution:
            case "normal":
                tempp_vect_rdn = np.random.normal(0, 1, vector_size)
                temp_list = tempp_vect_rdn.tolist()
            case "brown":   
                temp_list.append(-1 if random.random() < 0.5 else 1)
                for i in range(1, vector_size):
                    mov = -1 if random.random() < 0.5 else 1
                    val = temp_list[i-1] + mov + _RANDOM_WALK_NON_ZERO_DRIFT_VALUE
                    temp_list.append(val)
            case _:
                pass
    else:
        if distribution not in VALID_ARGUMENT_LIST_FOR_GEOPAIRING_DISTRIBUTION_NAME:
            return (ERROR_MESSAGE_SYS_EXIT_INVALID_ARGUMENT + "\"" + str(distribution) + "\"", FAILED, None)
    return (None, SUCCESS, np.array(temp_list))



def _env_stat():
    """
    @brief environment for statistic test and plot application
    @date: 2024 septembre 02
    """
    returnStruct_env_geoPairing = {"plt_obj": None}
    
    # Autocorrelation function comparison
    (_, _, data) = generate_random_vector(5000, distribution="brown")
    corr_1 = autocorrelation_1(data)
    corr_2 = autocorrelation_2(data)
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(data)
    ax[1,0].plot(corr_1)
    ax[0,1].plot(data)
    ax[1,1].plot(corr_2)
    
    # output struct construction for return
    returnStruct_env_geoPairing["plt_obj"] = plt
    return (None, SUCCESS, returnStruct_env_geoPairing)
    
    
    
def _env_geoPairing():
    """
    @brief: environment for geopairing application
    @date: 2024 August 30
    """
    returnStruct_env_geoPairing = {"plt_obj": None}
    
    stp_v = np.array([10, 100, 1000, 10000])
    virtual_step_time = 1e-3
    deviation_vector = list()
    dev_for_plot = list()
    
    for stp_elem in stp_v:
        fig, ax = plt.subplots()
        for i in range(1):
            data_dev = 0
            data_mean = 0
            (_, _, data) = generate_random_vector(_GEOPAIRING_RANDOM_VECTOR_LEN)
            ax.plot(data)
            for j in range(0, _GEOPAIRING_RANDOM_VECTOR_LEN-stp_elem, stp_elem):
                deviation_vector.append(data[j+stp_elem] - data[j])
            dev_vect_np = np.array(deviation_vector)
            u_dev = np.mean(dev_vect_np)
            std_dev = np.std(dev_vect_np)
            dev_for_plot.append(std_dev)
        ax.xaxis.grid()
        ax.yaxis.grid()
        fig.set_dpi(100)
    returnStruct_env_geoPairing["plt_obj"] = plt
    return (None, SUCCESS, returnStruct_env_geoPairing)



def __sys_arg_env_selection(input_arg):
    """
    @brief: select correct environment to launch for application execution
    @date: 2024 August 31
    """
    input_arg = input_arg[1:] # remove main.py function call
    for elem in input_arg:
        match elem:
            case "geo":
                return _env_geoPairing()
            case "stat":
                return _env_stat()
            case _:
                pass
    return (ERROR_MESSAGE_SYS_EXIT_NO_MATCH, FAILED, None)



def __sys_arg_verification(input_arg):
    """
    @brief: verify argument provided to prompt terminal during main execution
    @date: 2024 August 30
    """
    returnDict = {"flag_plot_required": False,
                  "flag_TODO": False}
    if not isinstance(input_arg, list):
        return (ERROR_MESSAGE_SYS_EXIT_UNEXPECTED_BEHAVIOUR, FAILED, None)
    for elem in input_arg:
        if elem not in VALID_ARGUMENT_LIST_FOR_SYS:
            return (ERROR_MESSAGE_SYS_EXIT_INVALID_ARGUMENT + "\"" + str(elem) + "\"", FAILED, None)
        if elem == "-p":
            returnDict["flag_plot_required"] = True
    return (None, SUCCESS, returnDict)



#########################################################
# MAIN CODE ENVIRONMENT FOR CODE AND FUNCTION EXECUTION #
#########################################################
if __name__ == "__main__":
    input_from_prompt = sys.argv
    (out_msg, out_flag, out__sys_arg_verification) = __sys_arg_verification(input_from_prompt)
    if (SUCCESS != out_flag):
        sys.exit(out_msg)
    (out_msg, out_flag, out_data_struct) = __sys_arg_env_selection(input_from_prompt)
    if (SUCCESS != out_flag):
        sys.exit(out_msg)
    if (out__sys_arg_verification["flag_plot_required"]):
        out_data_struct["plt_obj"].show()
    sys.exit(0)