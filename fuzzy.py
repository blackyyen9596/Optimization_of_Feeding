import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
# %%
# New Antecedent/Consequent objects hold universe variables and membership
# functions
feed_area = ctrl.Antecedent(np.arange(0, 50.2, 0.1), 'Area(cm2)')
quantity = ctrl.Antecedent(np.arange(0, 13.2, 0.1), 'Quantity')
grade = ctrl.Consequent(np.arange(0, 700.2, 0.1), 'feed(g)')


quantity['L'] = fuzz.trapmf(quantity.universe, [0, 0, 0, 5])
quantity['M'] = fuzz.trimf(quantity.universe, [0, 5, 10])
quantity['H'] = fuzz.trapmf(quantity.universe, [5, 10, 13, 13])

feed_area['S'] = fuzz.trapmf(feed_area.universe, [0, 0, 0, 20])
feed_area['M'] = fuzz.trimf(feed_area.universe, [0, 20, 40])
feed_area['L'] = fuzz.trapmf(feed_area.universe, [20, 40, 50, 50])

grade['N'] = fuzz.trapmf(grade.universe, [0, 0, 0, 300])
grade['L'] = fuzz.trimf(grade.universe, [0, 300, 600])
grade['A'] = fuzz.trapmf(grade.universe, [300, 600, 700, 700])


# view
feed_area.view()
quantity.view()
grade.view()

# %%
rule1 = ctrl.Rule(feed_area['S'] & quantity['L'], grade['N'])
rule2 = ctrl.Rule(feed_area['S'] & quantity['M'], grade['A'])
rule3 = ctrl.Rule(feed_area['S'] & quantity['H'], grade['A'])

rule4 = ctrl.Rule(feed_area['M'] & quantity['L'], grade['N'])
rule5 = ctrl.Rule(feed_area['M'] & quantity['M'], grade['L'])
rule6 = ctrl.Rule(feed_area['M'] & quantity['H'], grade['L'])

rule7 = ctrl.Rule(feed_area['L'] & quantity['L'], grade['N'])
rule8 = ctrl.Rule(feed_area['L'] & quantity['M'], grade['N'])
rule9 = ctrl.Rule(feed_area['L'] & quantity['H'], grade['N'])

grade_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,
                                 rule4, rule5, rule6,
                                 rule7, rule8, rule9])
grade_system = ctrl.ControlSystemSimulation(grade_ctrl)


# %%
grade_system.input['Area(cm2)'] = 22 #0-10
grade_system.input['Quantity'] = 1 #8-10
grade_system.compute()
# view
grade.view(sim=grade_system)
print(grade_system.output['feed(g)'])

plt.pause(0)