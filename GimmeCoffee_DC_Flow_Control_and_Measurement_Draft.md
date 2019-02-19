Lynn Li (ll643), Allison Tran (ant42), Yeonjin Yun (yy374)
Team GimmeCoffee

Researchers in the AguaClara laboratory collected the following head loss data through a 1/8" diameter tube that was 2 m long using water at 22°C. The data is in a comma separated data (.csv) file named Head_loss_vs_Flow_dosing_tube_data.csv. Use the pandas read csv function (pd.read_csv('filename.csv')) to read the data file. Display the data so you can see how it is formatted.
```python
import pandas as pd
import numpy as np
from aguaclara.core.units import unit_registry as u
from aguaclara.core import physchem as pc
from aguaclara.core import utility as ut
import matplotlib.pyplot as plt
import aguaclara.core.materials as mat
import aguaclara.core.constants as con
import aguaclara.core.utility as ut
import numpy as np
from scipy import interpolate, integrate

url = 'https://raw.githubusercontent.com/AguaClara/CEE4520/master/DC/data/Head_loss_vs_Flow_dosing_tube_data.csv'
head_loss_data = pd.read_csv(url)
array = np.array(head_loss_data)
head_loss_data
```
#Part 1
Using the data table above, assign the head loss and flow rate data to separate 1-D arrays. Attach the correct units. np.array can extract the data by simply inputting the text string of the column header. Name the array of flow rates Q_data. Here is example code to create the first array:
```python
HL_data=np.array(head_loss_data['Head loss (m)'])*u.m
print('There are',HL_data.size,'elements in this array.')

Q_data=np.array(head_loss_data['Flow rate (mL/min)'])*(u.mL/u.min)
print('There are',Q_data.size,'elements in this array.')
```
#Part 2
Calculate and report the maximum and minimum Reynolds number for this data set. Use the tube and temperature parameters specified above. Use the numpy min and max functions which take arrays as their inputs.
```python
PipeRough = 0*u.mm
L_tube = 2*u.m
T_data = 22 * u.degC
Nu_data = pc.viscosity_kinematic(T_data)
D_tube = 1/8*u.inch
Reynolds=pc.re_pipe(Q_data, D_tube, Nu_data)
Min=np.amin(Reynolds)
Max=np.amax(Reynolds)
print('The maximum Reynolds number for this data set is ',ut.round_sf(Max,3),'.')
print('The minimum Reynolds number for this data set is ',ut.round_sf(Min,3),'.')
```
#Part 3
You will now create a graph of head loss vs flow for the tube mentioned in the previous problems. This graph will have two sets of data: the real data contained within the csv file and a hydraulic model. The hydraulic model is what we would expect the head loss through the tube to be in an ideal world for any given flow. When calculating the hydraulic model head loss, assume that minor losses are negligible. Plot the data from the csv file as individual data points and the hydraulic model head loss as a continuous curve. Make the y-axis have units of cm and the x-axis have units of mL/s.
```python
HL_data.to(u.cm)
Q_data.to(u.mL/u.s)
plt.figure()
plt.plot(HL_data, Q_data, label = 'Real Data')
plt.xlabel('Flow Rate (mL/s)')
plt.ylabel('Head Loss (cm)')
plt.title('Flow Rate vs Head Loss')
plt.legend()
plt.show()
```
#Part 4
#Part 5
#Part 6
#Part 7
#Part 8
#Part 9
Could you use the original LFOM diameter by increasing the depth of the entrance tank by 10 or 20 cm? The new LFOM depth range would then be either 30 or 40 cm.
