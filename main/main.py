import numpy as np # Used for sigmoid only
import matplotlib
import matplotlib.pyplot as plt

### Constants

WEIGHTS = "weights_0.txt"
WEIGHTS_OUT = "weights1.txt"
DATASET = "datab2.txt"
TRAINS = 0
TESTS = 1000
P_STEP = 1 # Print every n loops, not 0
PLOT_FILE = "plot1000.png"
PLOT = False
EPOCHS = 1
LEARN_RATE = 0.5

### Weights from file

wt = []

with open(WEIGHTS, 'r') as file:
    for line in file.read().split("\n"):
        for i in line.split(" "):
          try:
            wt.append(float(i.strip()))
          except ValueError:
            print(i)

### Dataset from file

inp = []
answ = []

with open(DATASET, 'r') as file:
    for line in file.read().split("\n"):
      inp.append([])
      inp[-1].append(float(line.split(" ")[0]))
      inp[-1].append(float(line.split(" ")[1]))
      answ.append(float(line.split(" ")[2]))

###

# Neuron Class
class Nn():
  def __init__(self, in1, in2, w1, w2, b):

    self.in1 = in1
    self.in2 = in2
    self.w1 = w1
    self.w2 = w2
    self.b = b
    self.o = 0

  def nsum(self):
    self.o = float(self.in1*self.w1+self.in2*self.w2+self.b)

  def sigm(self):
    self.o = float(1/(1+np.exp(-self.o)))

# Run class
class Train():

  def run(self, inp, wt, i):
    self.n1 = Nn(inp[i][0], inp[i][1], wt[0], wt[1], wt[2])
    self.n1.nsum()
    self.n2 = Nn(inp[i][0], inp[i][1], wt[3], wt[4], wt[5])
    self.n2.nsum()
    self.o1 = Nn(self.n1.o, self.n2.o, wt[6], wt[7], wt[8])
    self.o1.nsum()
    self.o1.sigm()
    return self.o1.o

# Adjust weights **(In Progress)**
def adjust(inp, ans, i):

  global LEARN_RATE
  
  output = Train()
  
  for j in range(9):
  
    d_slope = []
  
    output.run(inp, wt, i)
  
    # N1
    d_slope.append(w1_slope(output.o1.o, ans, output.o1.w1, inp[i][0]))
    d_slope.append(w2_slope(output.o1.o, ans, output.o1.w2, inp[i][0]))
    #d_slope.append(b1_slope(output.o1.o, ans, output.o1.w1))
    d_slope.append(0)
  
    # N2
    d_slope.append(w3_slope(output.o1.o, ans, output.o1.w1, inp[i][1]))
    d_slope.append(w4_slope(output.o1.o, ans, output.o1.w2, inp[i][1]))
    #d_slope.append(b2_slope(output.o1.o, ans, output.n2.w2))
    d_slope.append(0)
  
    # O1
    d_slope.append(w5_slope(output.o1.o, ans, output.n1.o))
    d_slope.append(w6_slope(output.o1.o, ans, output.n2.o))
    #d_slope.append(b3_slope(output.o1.o, ans))
    d_slope.append(0)
  
    wt[j] += -LEARN_RATE * d_slope[j]

def cost(prediction, target):
  return (prediction - target)**2

def cost_slope(prediction, target):
  return 2 * (prediction - target)

def b1_slope(o,a,w5):
  return 2 * (o-a) * o * (1-o) * w5

def b2_slope(o,a,w4):
  return 2 * (o-a) * o * (1-o) * w4

def b3_slope(o,a):
  return 2 * (o-a) * o * (1-o)

def w1_slope(o,a,w5,x):
  return 2 * (o-a) * o * (1-o) * w5 * x

def w2_slope(o,a,w6,x):
  return 2 * (o-a) * o * (1-o) * w6 * x

def w3_slope(o,a,w5,y):
  return 2 * (o-a) * o * (1-o) * w5 * y

def w4_slope(o,a,w6,y):
  return 2 * (o-a) * o * (1-o) * w6 * y

def w5_slope(o,a,n1):
  return 2 * (o-a) * o * (1-o) * n1

def w6_slope(o,a,n2):
  return 2 * (o-a) * o * (1-o) * n2

###

outp = [] # Outputs
ts_outp = [] # Test Outputs
t_cost = [] # The costs of each loop (Train costs)
ts_cost = [] # Test costs
total_err = 0
ts_total_err = 0

###

# Running loop
for epo in range(EPOCHS):
  epo += 1
  pref = ("  " if (epo % 2) == 0 else "")

  print("#"*10)

  # Training
  for i in range(TRAINS):

    output = Train()

    outp.append(output.run(inp, wt, i))
    t_cost.append(cost(outp[-1], answ[i]))
    
    adjust(inp, answ[i], i)

    total_err += cost(ts_outp[-1], answ[i])

    if (i + 1 )% P_STEP == 0:
      print("{0}{1}:{2}=>{3:.3f}:{4}".format(pref,epo,i+1,outp[-1],answ[i]))
      
  print("{0}{1}".format(pref,"#"*10))

  # Testing
  for i in range(TESTS):
    i += TRAINS

    output = Train()

    ts_outp.append(output.run(inp, wt, i))
    ts_cost.append(cost(ts_outp[-1], answ[i]))

    ts_total_err += cost(ts_outp[-1], answ[i])

    if (i + 1 )% P_STEP == 0:
      print("{0}{1}:{2}=>{3:.3f}:{4}".format(pref,epo,i+1,ts_outp[-1],answ[i]))

### Total error

print("#"*10)

print("Total Error")
print("Training {0}/{1}".format(str(total_err), TRAINS))
print("Testing {0}/{1}".format(str(ts_total_err), TESTS))

### Outputing last weights to file

with open(WEIGHTS_OUT, 'w') as file:
  fileout = ''
  for j, i in enumerate(wt):
    fileout += str(i) + ' '
    if ((j + 1) % 3) == 0:
      fileout += '\n'
  file.write(fileout)

### Ploting dataset on graph

if PLOT:
  matplotlib.rcParams['axes.unicode_minus'] = False
  fig, ax = plt.subplots()
  b_color = [0,0,1]
  r_color = [1,0,0]
  g_color = [0,1,0] # Identical
  for i in inp:
    if i[0] < i[1]:
      colr = r_color
    else:
      if i[0] == i[1]:
        colr = g_color
      else:
        colr = b_color
    ax.plot(i[0], i[1], 'o', color=colr)
  ax.set_title('Flower Height by Width')

  plt.savefig(PLOT_FILE)

  ###