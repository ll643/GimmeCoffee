# Flocculator Design DC
### Team GimmeCoffee
#### Lynn Li (ll643), Allison Tran (ant42), and Yeonjin Yun (yy374)

## Velocity gradients and flow geometry
### 1)

Coagulant is injected in the center a long straight pipe. The pipe is 12 inches Nominal Diameter schedule 40 PVC and the flow rate is 120 L/s at $10^{\circ}C$. What distance is required for the flow to be completely mixed? Note that this estimate is based on the time required for an eddy to traverse the diameter of the pipe and that a safety factor of order 3 * pi/2 would be reasonable. Include this safety factor in the calculations. See the (equation for pipe mixing)[https://aguaclara.github.io/Textbook/Rapid_Mix/RM_Derivations.html?highlight=energy%20dissipation#equation-rapid-mix-rm-derivations-42]?
```python
from aguaclara.core.units import unit_registry as u
from aguaclara.core import physchem as pc
from aguaclara.core import pipes as pipes
from aguaclara.core import utility as ut
from aguaclara.core import constants as con
from aguaclara.research import floc_model as fm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from aguaclara.core import materials as mat

Nominal_Diameter = (12*u.inch).to(u.m)
Temp = (10*u.celsius).to(u.kelvin)
Q = 120*(u.L/u.sec)
Safety_Factor = (3*math.pi)/2
Nu = pc.viscosity_kinematic(Temp)
Inner_Diameter = pipes.ID_sch40(Nominal_Diameter)
PipeRough = mat.PVC_PIPE_ROUGH
f = pc.fric(Q, Inner_Diameter, Nu, PipeRough)
N_D_Pipe = (2/f)**(1/3)
Distance = N_D_Pipe * Safety_Factor * Nominal_Diameter
Distance

print('The distance required for the flow to be completely mixed is',ut.round_sf(Distance,3))
```

### 2)

What is the residence time in this mixing zone?

```python
Reynum=pc.re_pipe(Q,Inner_Diameter,Nu)
V_bar=Reynum*Nu/(Inner_Diameter.to(u.m))
#V_eddy = (PipeRough * Nominal_Diameter)**(1/3)
#V_bar = V_eddy/N_D_Pipe
Residence_time = Distance/V_bar

Residence_time

print('The residence time is',ut.round_sf(Residence_time,3))
```
### 3)

How much head loss from wall shear will have occurred in the pipe in the distance measured in the previous problem? Compare this with the
```python
Head_Loss = (pc.headloss_fric(Q, Inner_Diameter, Distance, Nu, PipeRough)).to(u.cm)
Head_Loss

print('The head loss is',ut.round_sf(Head_Loss,3), '.')
```
### 4)

What is the Camp Stein velocity gradient in this pipe flow?
```python
G_cs = ((((32*f)/(Nu*(np.pi**3)))*(((Q.to(u.m**3/u.s))**3)/((Inner_Diameter.to(u.m))**7)))**(1/2)).to(u.Hz)
G_cs
print('The Camp Stein velocity gradient is',ut.round_sf(G_cs,3), '.')
```
### 5)

What is the $G\theta$ for this mixing zone and how does it compare with the  $G\theta$ recommended for [mechanical mixing units](https://aguaclara.github.io/Textbook/Rapid_Mix/RM_Intro.html#maximum-velocity-gradients)?

```python
Mix_HRT = np.array([0.5,15,25,35,85])*u.s
Mix_G = np.array([4000,1500,950,850,750])/u.s
Mix_Gt = np.multiply(Mix_HRT, Mix_G)

Mix_Gt_exp = G_cs*Residence_time
Mix_Gt_exp
print('The $G\Theta$ for this mixing zone is',ut.round_sf(Mix_Gt_exp,3), '.')
```
Our experimental $G\Theta$ is lower than that of the recommended $G\Theta$ in the lab manual.
### 6)

What is the velocity gradient at the wall of the pipe? This will make it apparent that the velocity gradient is far from constant

```python
G_wall = ((f)*(V_bar**2))/(8*Nu)
G_wall
print('The velocity gradient at the wall of the pipe is',ut.round_sf(G_wall,3),', which is far from',ut.round_sf(G_cs,3),'. Therefore, velocity gradient is not a constant.')

```
### 7)

Suppose we insert a flat plate oriented with the flat surface facing the flow inside the pipe. Let the width of the plate be 0.5 cm so it is small enough that it doesn't significantly increase the velocity in the pipe. What is the maximum velocity gradient downstream of the plate? You may neglect the fact that the velocity in the center of the turbulent pipe flow is slightly higher than the average velocity.

$\varepsilon _{Max} = \Pi_{Plate}\frac{\bar v^3}{W_{Plate}}$

$G_{Max} = \bar v\sqrt{\frac{\Pi_{Plate} \bar v}{\nu W_{Plate}}}$
```python
width = (0.5*u.cm).to(u.m)
pi_plate = 0.04
G_Max = (V_bar)*(((pi_plate*V_bar)/(Nu*width))**(1/2)).to_base_units()
G_Max
print('The maximum velocity gradient downstream of the plate is',ut.round_sf(G_Max,3))
```
### 8)
What happens to the velocity gradient if a narrower flat plate is used? Does the maximum velocity gradient increase or decrease? Just look at the equation to answer this!

If a narrower flat plate is used, the velocity gradient would increase because the width of the plate is inversely proportionally to the velocity gradient, as seen in the equation used in the previous problem.

## Flocculation model

### 1)
How far will two kaolin clay particles (density of 2650 $\frac{kg}{m^3}$) with a diameter of 5 $\mu m$ travel relative to each if they are in a uniform velocity gradient of 100 Hz for 400 s and separated (in the direction of the velocity gradient) by their average separation distance based on a turbidity of 0.5 NTU? (We have defined NTU as a unit based on the concentration of clay in the aguaclara distribution. Note that in a uniform velocity gradient $\bar G = G_{CS}$.

 ```python
import numpy as np
from aguaclara.core.units import unit_registry as u
from aguaclara.core import physchem as pc, utility as ut

u.enable_contexts('chem')

density = 2650*(u.kg/u.m**3)
diameter = (5*u.micrometer)
G = 100*u.Hz
time = 400*u.sec
c_clay = 0.5*u.NTU

lambda_clay = ((density/c_clay)*((np.pi*diameter**3)/6))**(1/3)

velocity = lambda_clay * G
distance = velocity * time
distance = distance.to_base_units()

print('The separation distance is' ,ut.round_sf(lambda_clay.to(u.mm),3), '.')
print('The distance traveled relative to each other is',ut.round_sf(distance,2),'.')

```
### 2)

How much volume is "cleared" by these particles divided by the volume occupied by the particles? This ratio is essentially how many times these particles should have collided in the 400 s.
```python
V_cleared = (math.pi*(diameter**2)*lambda_clay*G*time).to_base_units()
V_cleared = V_cleared.to(u.uL)

volume_particles = (lambda_clay)**3

ratio = V_cleared/volume_particles
ratio = ratio.to_base_units()

print('The ratio between the volume cleared and the volume occupied is',ut.round_sf(ratio,3))

```
The above calculations illustrate why 1 NTU is a practical limit for flocculation. Assuming that we don't want to apply so much coagulant that the clay particles are completely covered with coagulant, then some fraction of the collisions will be ineffective. Thus at 1 NTU a $G \theta$ of 40,000 might only cause one successful collisions

## Flocculator design

Below we design a flocculator using aguaclara.design.floc in the aguaclara distribution version 0.0.20 (minimum). We will use the default settings for this design except change the flow rate to 60 L/s. The available inputs that you can change are:

* Q=20 * u.L/u.s,
* temp=25 * u.degC,
* max_L=6 * u.m,
* Gt=37000,
* HL = 40 * u.cm,
* downstream_H = 2 * u.m,
* ent_tank_L=1.5 * u.m,
* max_W=42 * u.inch,
* drain_t=30 * u.min

You can set any of these parameters by including those keywords in the function call.
```
myF = floc.Flocculator(Q=60 * u.L/u.s,temp = 0 * u.degC)

```
Below is our design
``` python
import aguaclara
import aguaclara.core.physchem as pc
import aguaclara.core.head_loss as minorloss
import aguaclara.core.pipes as pipes
import aguaclara.design.human_access as ha
from aguaclara.core.units import unit_registry as u
import numpy as np
import aguaclara.design.floc as floc
Q=60 * u.L/u.s
myF = floc.Flocculator(Q=Q,temp=0*u.degC)

# you can either access individual parameters
print('The number of channels is', myF.channel_n)
print('The channel length is',myF.channel_L)
print('The channel width is',ut.round_sf(myF.channel_W,2))
print('The spacing between baffles is',ut.round_sf(myF.baffle_S,2))
print('The number of obstacles per baffle is', myF.obstacle_n)
print('The velocity gradient is', ut.round_sf(myF.vel_grad_avg,2))
print('The residence time used for design is',ut.round_sf(myF.retention_time,2))
print('The maximum distance between flow expansions is', ut.round_sf(myF.expansion_max_H,2))
print('The drain diameter is', myF.drain_ND)
# or you can see the entire design as a dictionary of values
myF.design
```

## Calculations and analysis

### 1)

How many expansions are there in total? Estimate this based on the spacing and flocculator size. You will have to account for the entrance tank that occupies volume in the first flocculator channel.

```Python
Length = myF.channel_L
Width = ut.round_sf(myF.channel_W,2)
Ent_Length = myF.ent_tank_L
Area_Ent_Tank = Ent_Length*Width
Spacing_Area = (ut.round_sf(myF.baffle_S,2)).to(u.m)*Width
Area = Length*Width
Total_Area = (Area*2)-Area_Ent_Tank
baffles = Total_Area/Spacing_Area
Expansions = np.floor(baffles*myF.expansion_n)
Expansions
print('The number of expansions in total is',ut.round_sf(Expansions,3),'.')
```
    Expansions should be a dimensionless number yours has the unit of length. Also make sure to take into account the entrance tank being in the flocculator

### 2)
What is the head loss per expansion? (Calculate this head loss using the minor loss equation) You can use the BAFFLE_K that is defined in the flocculator class.
```python
import math
K = floc.Flocculator.BAFFLE_K
g = constants.GRAVITY
A = Width*(ut.round_sf(myF.baffle_S,2))
V = Q/A
Head_Loss = (K*(V**2)/(2*g))
Head_Loss = Head_Loss.to(u.mm)

print('The head loss per expansion is',ut.round_sf(Head_Loss,3),'.')
```

### 3)
What is the total head loss of all of the expansions? Compare this with the target head loss of 40 cm.
```Python
Total_Head_Loss = Head_Loss*Expansions
print('The total head loss per expansion is',ut.round_sf(Total_Head_Loss.to(u.cm),3),'.')
```
### 4)
Change the design temperature over a range that would be applicable in Ithaca (0 to 30 degC) for a flocculator design of your choice. What happens as the temperature increases? Plot residence time, velocity gradient, baffle spacing, and number of channels as functions of temperature. Explain WHY these design changes occur.

```Python
import matplotlib.pyplot as plt

Temp = ((np.arange(0,33,3))*u.degC)
Residence_time = np.zeros(11)
Velocity_gradient = np.zeros(11)
Baffle_Spacing = np.zeros(11)
Channel_Num = np.zeros(11)

for i in range(Temp.size):
  Q = 60 * u.L/u.s
  myF = floc.Flocculator( Q=Q, temp = Temp[i])
  Residence_time[i] = (ut.round_sf(myF.retention_time,2)).magnitude
  Velocity_gradient[i] = (ut.round_sf(myF.vel_grad_avg,2)).magnitude
  Baffle_Spacing[i] = (ut.round_sf(myF.baffle_S,2)).magnitude
  Channel_Num[i] = (ut.round_sf(myF.channel_n,2)).magnitude

Residence_time = Residence_time*u.s
Velocity_gradient = Velocity_gradient*(u.Hz)
Baffle_Spacing = Baffle_Spacing*u.cm

#Residence Time vs Temperature
plt.plot(Temp, Residence_time)
plt.xlabel('Temperature (K)')
plt.ylabel('Residence Time (seconds)')
plt.title('Residence Time vs Temperature')
plt.show()

#Velocity Gradient vs Temperature
plt.plot(Temp, Velocity_gradient)
plt.xlabel('Temperature (K)')
plt.ylabel('Velocity Gradient (Hertz)')
plt.title('Velocity Gradient vs Temperature')
plt.show()

#Baffle Spacing vs Temperature
plt.plot(Temp, Baffle_Spacing,'o')
plt.xlabel('Temperature (K)')
plt.ylabel('Baffle Spacing (cm)')
plt.title('Baffle Spacing vs Temperature')
plt.show()

#Number of Channels vs Temperature
plt.plot(Temp, Channel_Num,'o')
plt.xlabel('Temperature (K)')
plt.ylabel('Number of Channels (dimensionless)')
plt.title('Number of Channels vs Temperature')
plt.show()

```

The water becomes more viscous as it gets colder. Thus it becomes more difficult to deform. Given that we are limiting the amount of energy that we are willing to use, we have to compensate by deforming the fluid more slowly. Thus if we hold the amount of energy available as a constant, then the velocity gradient decreases as the temperature decreases and the residence time increases. The number of channels increases as the temperature drops because the design ran up against the maximum channel width constraint as the flocculator volume increased.

### 5)
When designing a flocculator how should you select the design temperature?

When designing a flocculator, you should select the design temperature based off of the lowest expected temperature for the design temperature so that your flocculator is still functional in case of extreme weather.


### 6)
We have been experimenting with flocculators that have a $G\theta$ of 20,000 and a head loss of 50 cm for use in Honduras where the minimum temperature is about 15 $^\circ C$. Create a design with these inputs and flow rates of 10 L/s and 60 L/s. To simplify the design, set the entrance tank length to zero. Note that our flocculator design algorithm currently requires there to be an even number of channels for the flocculator. The minimum width of the channels is 45 cm and thus when the flocculator volume is small the length of the flocculator is reduced.
Given these designs and given that our sedimentation tanks end up being closer to 7 m long, would you recommend that we change our plant layout to allow a single channel flocculator?
List as many design implications as you can think of for this potential change. Check out the [current cad drawing](https://cad.onshape.com/documents/5a7585ae3248902548b02541/w/349594d2eb30a283f019807e/e/add8912cf760c28f462bd04f) and identify what else would need to change if the flocculator only had one channel.


```Python
#Flocculator 1
Q = 60 * u.L/u.s
Gt = 20000
temp = 15 * u.degC
HL = 50*u.cm
myF_60= floc.Flocculator(Q=Q, Gt=Gt, temp=temp, HL=HL)
myF_60.ent_tank_L = 0*u.m
myF_60.design

#Flocculator 2
Q = 10 * u.L/u.s
Gt = 20000
temp = 15 * u.degC
HL = 50*u.cm
myF_10= floc.Flocculator(Q=Q, Gt=Gt, temp=temp, HL=HL)
myF_10.ent_tank_L = 0*u.m
myF_10.ent_tank_L
myF_10.design
```
No, we would not recommend a single channel flocculator. This is due to the fact that the flocculator requires a level of symmetry. After passing through the channels, the water needs to return to side where it originated from so that it can pass through the filter entrance tank (filter box). An odd number of channels would prevent this from happening. There are currently no solutions which would allow us to move the water from one side to another without the addition of another channel. In addition, if our sedimentation tank design ended up being close to 7m long, we wouldn't even be able to purchase a pipe of that length. The type of pipe required is only manufactured at around 6.5m.
