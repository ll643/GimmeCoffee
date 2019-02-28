# Flow Control and Measurement Design Challenge
##Team GimmeCoffee
Lynn Li (ll643), Allison Tran (ant42), Yeonjin Yun (yy374)


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
### 1)
Using the data table above, assign the head loss and flow rate data to separate 1-D arrays. Attach the correct units. np.array can extract the data by simply inputting the text string of the column header. Name the array of flow rates Q_data. Here is example code to create the first array:
```python
HL_data=np.array(head_loss_data['Head loss (m)'])*u.m
print('There are',HL_data.size,'elements in this array.')

Q_data=np.array(head_loss_data['Flow rate (mL/min)'])*(u.mL/u.min)
print('There are',Q_data.size,'elements in this array.')
```
### 2)
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
### 3)
You will now create a graph of head loss vs flow for the tube mentioned in the previous problems. This graph will have two sets of data: the real data contained within the csv file and a hydraulic model. The hydraulic model is what we would expect the head loss through the tube to be in an ideal world for any given flow. When calculating the hydraulic model head loss, assume that minor losses are negligible. Plot the data from the csv file as individual data points and the hydraulic model head loss as a continuous curve. Make the y-axis have units of cm and the x-axis have units of mL/s.
```python
from scipy.optimize import curve_fit
flow_values= np.linspace(41.83/148,1,50) * 148 *(u.mL/u.minute)

HL= np.zeros(50) * pc.headloss_fric(Q_data[0],D_tube,L_tube,Nu_data,PipeRough).units

for i in range (len(HL)):
  HL[i]= pc.headloss_fric(flow_values[i],D_tube,L_tube,Nu_data,PipeRough)

HL_data.to(u.cm)
Q_data.to(u.mL/u.s)
flow_values.to(u.mL/u.s)

plt.figure()
plt.plot(HL_data, Q_data, label = 'Real Data')
plt.plot(HL, flow_values, label = 'Theoretical Data')
plt.xlabel('Flow Rate (mL/s)')
plt.ylabel('Head Loss (cm)')
plt.title('Flow Rate vs Head Loss')
plt.legend()
plt.show()

def HL_curvefit(FlowRate, KMinor):
  HL_curvefit = np.zeros(FlowRate.size) *pc.headloss_fric(FlowRate[0], D_tube, L_tube, Nu_data, PipeRough).units
  for i in range(FlowRate.size):
      HL_curvefit[i] = pc.headloss(FlowRate[i], D_tube, L_tube, Nu_data, PipeRough, KMinor)
  return HL_curvefit.magnitude

popt, pcov = curve_fit(HL_curvefit, Q_data, HL_data, bounds=[[0.],[20]])
K_minor_fit = popt[0]
print('The best fit minor loss coefficient is',ut.round_sf(K_minor_fit,2))
plt.plot(Q_data.to(u.mL/u.s), ((HL_curvefit(Q_data, *popt))*u.m).to(u.cm), 'r-', label='Minor Loss')
plt.xlabel('Flow Rate (mL/s)')
plt.ylabel('Head Loss (cm)')
plt.title('Flow Rate vs Head Loss')
plt.legend()
plt.show()

RMSE_Kminor = (np.sqrt(np.var(np.subtract((HL_curvefit(Q_data, *popt)),HL_data.magnitude)))*u.m).to(u.cm)
print('The root mean square error for the model fit when adjusting the minor loss coefficient was',ut.round_sf(RMSE_Kminor,2))
```
### 4)
Repeat the analysis above, but this time assume that the minor loss coefficient is zero and that diameter is the unknown parameter. The bounds specified in the line beginning with popt, pcov should be changed from the previous question (which had bounds from 0 to 20) to the new bounds of 0.001 to 0.01.

Hint: You only need to change the name of the defined function (perhaps "HL_curvefit2"?) and adjust its inputs/values.

```python
def HL_curvefit2(FlowRate, D_tube):
  HL_curvefit2 = np.zeros(FlowRate.size) *pc.headloss_fric(FlowRate[0], D_tube, L_tube, Nu_data, PipeRough).units
  for i in range(FlowRate.size):
      HL_curvefit2[i] = pc.headloss(FlowRate[i], D_tube, L_tube, Nu_data, PipeRough, KMinor)
  return HL_curvefit2.magnitude

KMinor= 0
popt, pcov = curve_fit(HL_curvefit2, Q_data, HL_data, bounds=[[0.001],[0.01]])
plt.plot(Q_data.to(u.mL/u.s), ((HL_curvefit2(Q_data, *popt))*u.m).to(u.cm), 'bo', label='No Minor Loss')
plt.plot(Q_data.to(u.mL/u.s), ((HL_curvefit(Q_data, *popt))*u.m).to(u.cm), 'r-', label='Minor Loss')
plt.xlabel('Flow Rate (mL/s)')
plt.ylabel('Head Loss (cm)')
plt.title('Flow Rate vs Head Loss')
plt.legend()
plt.show()

RMSE_Kminor = (np.sqrt(np.var(np.subtract((HL_curvefit2(Q_data, *popt)),HL_data.magnitude)))*u.m).to(u.cm)

print('The root mean square error for the model fit when adjusting the minor loss coefficient was',ut.round_sf(RMSE_Kminor,2))
```

### 5)
Changes to which of the two parameters, minor loss coefficient or tube diameter, results in a better fit to the data?

Changes to minor loss resulted in a better fit to the data. The root mean square error of a varied minor loss was smaller than that of a varied tube diameter.

### 6)
Create a design for a chemical dose controller using aguaclara. Use the AguaClara cdc functions to obtain the diameter, length, and number of dosing tubes that are required. Then take that design and use the physchem functions for flow in a pipe to calculate the maximum coagulant dose that it will deliver. This design and check is a powerful tool to ensure that you don't make mistakes because the design and the check are done using different code.

I've included the code necessary to get the length of the dosing tubes. Note that we are currently specifying the diameter of the dosing tubes to be 1/8".

```python
from aguaclara.core.units import unit_registry as u
from aguaclara.design import cdc as cdc
from aguaclara.core import physchem as pc
import numpy as np
FlowPlant = 60 * u.L/u.s
ConcDoseMax = 20 * u.mg/u.L
ConcStock = 60 * u.g/u.L
DiamTubeAvail = np.array([1/8]) * u.inch
HeadlossCDC = 20 * u.cm
LenCDCTubeMax = 6* u.m
temp = 20 * u.degC
# 1 is for PACl
en_chem =  1
KMinor = 2

L_tube = cdc.len_cdc_tube(FlowPlant, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax, temp, en_chem, KMinor)
print('The length of the dosing tube is', ut.round_sf(L_tube,2))

Nu_pacl=cdc.viscosity_kinematic_pacl(60, temp)

DiamTube=cdc.diam_cdc_tube(FlowPlant, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax,temp, en_chem, KMinor)


FlowPipe=pc.flow_pipe(DiamTube, HeadlossCDC, L_tube, Nu_pacl, PipeRough, KMinor).to(u.liter/u.s)

Number_Tubes = cdc.n_cdc_tube(FlowPlant,ConcDoseMax, ConcStock,DiamTubeAvail, HeadlossCDC, LenCDCTubeMax,temp, en_chem, KMinor)


Total_Dose= (Number_Tubes*FlowPipe*ConcStock/FlowPlant).to(u.milligram/u.liter)

print('The maximum coagulant dose is ',ut.round_sf(Total_Dose,2))
print('The number of tubes needed is ',ut.round_sf(Number_Tubes,1))
print('The diameter of the tube is ',ut.round_sf(DiamTube.to(u.inch),3))
```

### 7)
An AguaClara plant will be upgraded from 20 L/s to 30 L/s by adding two sedimentation tanks, increasing the head loss through the flocculator, and adding an additional StaRS filter. Give the current design specs for the CDC. Propose a simple modification to the CDC to handle the additional flow.

```python
FlowPlant20= 20 * u.L/u.s

DiamTube20=cdc.diam_cdc_tube(FlowPlant20, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax, temp, en_chem, KMinor)

L_tube20= cdc.len_cdc_tube(FlowPlant20, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax, temp, en_chem, KMinor)

Number_Tubes20 = cdc.n_cdc_tube(FlowPlant20,ConcDoseMax, ConcStock,DiamTubeAvail, HeadlossCDC, LenCDCTubeMax,temp, en_chem, KMinor)


print('For 20L/s plant, the diameter of dosing tube is ',DiamTube20.to(u.inch),', the length of the dosing tube is ',ut.round_sf(L_tube20,2),'and the number of the dosing tube is ',ut.round_sf(Number_Tubes20,1),'.')

FlowPlant30= 30 * u.L/u.s

DiamTube30=cdc.diam_cdc_tube(FlowPlant30, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax, temp, en_chem, KMinor)

L_tube30= cdc.len_cdc_tube(FlowPlant30, ConcDoseMax, ConcStock, DiamTubeAvail, HeadlossCDC, LenCDCTubeMax, temp, en_chem, KMinor)

Number_Tubes30 = cdc.n_cdc_tube(FlowPlant30,ConcDoseMax, ConcStock,DiamTubeAvail, HeadlossCDC, LenCDCTubeMax,temp, en_chem, KMinor)


print('For 30L/s plant, the diameter of dosing tube is ',DiamTube30.to(u.inch),', the length of the dosing tube is ',ut.round_sf(L_tube30,2),'and the number of the dosing tube is ',ut.round_sf(Number_Tubes30,1),'.')
```
The number of the tubes increased from 2 to 3.

We applied the design upgrades to the previous model for the CDC, and found that these changes caused the coagulant dose to decrease.


### 8)
The LFOM for the 20 L/s plant was designed to have a safety factor of 1.2 and the entrance tank water level changes by 20 cm as the flow goes from zero to 20 L/s. The LFOM for this design is an 8 inch SDR 26 pipe. Determine if the LFOM diameter will need to be increased to handle a flow of 30 L/s.

```python
import aguaclara.design.lfom as lfom
mylfom = lfom.LFOM(safety_factor=1.2)
mylfom20 = lfom.LFOM(20 * u.L/u.s,safety_factor=1.2)
mylfom20.nom_diam_pipe
print('The LFOM for the 20L/s plant has a diameter of',ut.round_sf(mylfom20.nom_diam_pipe,2),'.')
#8 in

mylfom = lfom.LFOM(safety_factor=1.2)
mylfom30 = lfom.LFOM(30 * u.L/u.s,safety_factor=1.2)
mylfom30.nom_diam_pipe
print('The LFOM for the 30L/s plant has a diameter of',ut.round_sf(mylfom30.nom_diam_pipe,2),'.')
#10 in

print('The LFOM diameter will need to be increased from',mylfom20.nom_diam_pipe ,'to ',mylfom30.nom_diam_pipe ,'to handle a flow of 30 L/s, assuming that safety factor stays the same.')
```
### 9)
Could you use the original LFOM diameter by increasing the depth of the entrance tank by 10 or 20 cm? The new LFOM depth range would then be either 30 or 40 cm.

```python
mylfom = lfom.LFOM(safety_factor=1.2)

newLFOM1=lfom.LFOM(q=30*u.L/u.s,hl=30*u.cm,safety_factor=1.2)
newLFOM1.nom_diam_pipe

newLFOM2=lfom.LFOM(q=30*u.L/u.s,hl=40*u.cm,safety_factor=1.2)
newLFOM2.nom_diam_pipe

print('The original diameter was 8 inch. The new LFOM depth range of 40cm works with the original diameter, but the depth range of 30cm does not work with the original diameter.')
```
