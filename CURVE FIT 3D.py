import scipy.optimize as optimize
import numpy as np

#format xx.xx
real_param = [6.23, 5.11, 1, 1, 1, 1]
# the bound of the area we explore
x_bound=2
y_bound = 3


# [a,b] [a',b'] noise%
def is_param_equal(real_param, new_param, noise):
    for i in len(real_param):
        if abs(real_param[i] - new_param[i]) >= real_param[i] * noise:
            return False
    return True


def noise_vector(real_param, new_param):
    res=[]
    for i in range(len(real_param)):
        res.append(round(abs(real_param[i]-new_param[i]),2))
    return res


# input: [a,b] [a',b'] - assume same len
# output: [a,b] [a',b']

#noise measures
#average abs differnce
def noise_(real_param, new_param):
    return sum(noise_vector(real_param, new_param))/len(real_param)


#אפשר לחשוב על מדדים נוספים- בפרט אחוזים- מנורמל ביחס לכל פרמטר ועוד

def original_func(x, y, noise=0.0):
    noise = np.random.normal(loc=0, scale=noise)
    global real_param
    a, b, c, d, e, f = real_param[0], real_param[1], real_param[2], real_param[3], real_param[4], real_param[5]
    return func_template(x, y, a, b, c, d, e, f)+noise

def func(data, a, b, c, d, e, f):
    #return (data[:, 0] * data[:, 0] * a) + (data[:, 1] * data[:, 1] * b)
    return func_template(data[:, 0], data[:, 1], a, b, c, d, e, f)

def func_template(x, y, a, b, c=0, d=0, e=0, f=0):
    return (a*(x-c)*(x-c)+b*(y-d)*(y-d)) *np.exp(-e* x ** 2 - f* y ** 2)


def create_samples(num_of_samples=10, noise=0):
    # rand
    A = []
    for i in range(num_of_samples):
        x= np.round(np.random.uniform(-1*x_bound, x_bound),2)
        y= np.round(np.random.uniform(-1*y_bound, y_bound),2)
        z= original_func(x, y, noise)
        # print("Sample "+str(i+1)+": "+str((x, y, z)))
        A.append((x, y, z))
    #A = np.array([(19, 20, np.round(np.random.uniform((-1*y_bound, y_bound)),2)), (10, 40, original_func(10, 40)), (20, 40, original_func(20, 40))])
    A = np.array(A)
    return A

#quality_measurment
def the_everage_distance(real_param, params_list):
    num_of_samples=100
    sum=0
    for i in range(num_of_samples):
        x = np.round(np.random.uniform(-1 * x_bound, x_bound), 2)
        y = np.round(np.random.uniform(-1 * y_bound, y_bound), 2)
        z = func_template(x, y, real_param[0], real_param[1], real_param[2], real_param[3], real_param[4], real_param[5])
        z_tag = func_template(x, y, params_list[0], params_list[1], params_list[2], params_list[3], params_list[4],
                          params_list[5])
        sum+=abs(z_tag-z)
        # print("Sample "+str(i+1)+": "+str((x, y, z)))
    return sum/num_of_samples



guess = (3, 5, 2, 3, 2, 1)
num_of_samples=7
noise=0.04
print("noise  "+str(noise))





num_of_samples=7
print("num_of_samples"+str(num_of_samples))
A = create_samples(num_of_samples, noise)
params, pcov = optimize.curve_fit(func, A[:, :2], A[:, 2], guess)
params_list = params.tolist()
params_list = [round(params_list[0], 2), round(params_list[1], 2), round(params_list[2], 2),
               round(params_list[3], 2), round(params_list[4], 2), round(params_list[5], 2)]
if params_list == real_param:
    print("num of samples")
    print(num_of_samples)
    print("The real param recenstructed well")
print("The real param")
print(real_param)
print("The recustruced  param")
print(params_list)
print(the_everage_distance(real_param,params_list))




