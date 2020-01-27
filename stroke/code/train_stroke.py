# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
# read train and test data
train_data = pd.read_csv('../data/train_2v.csv')
train_data = train_data.drop(['id'], axis=1)

test_data = pd.read_csv('../data/test_2v.csv')
test_data = test_data.drop(['id'], axis=1)

# %%
# do some data analysis before training and testing data
stroke_data = train_data[train_data['stroke'] == 1]
unstroke_data = train_data[train_data['stroke'] == 0]

# %%
# gender vs stroke
genderChart(stroke_data, unstroke_data)
# age vs stroke
ageChart(stroke_data, unstroke_data)
# hypertension vs age
hypertensionChart(stroke_data, unstroke_data)
# heart_disease vs stroke
heartDiseaseChart(stroke_data, unstroke_data)
# ever_married vs stroke
marriageChart(stroke_data, unstroke_data)
# work_type vs stroke
workTypeChart(stroke_data, unstroke_data)
# residence_type vs stroke
residecTypeChart(stroke_data, unstroke_data)
# agv_glucose vs stroke
agvGlucoseChart(stroke_data, unstroke_data)
# bmi vs stroke
bmiChart(stroke_data, unstroke_data)
# smoking_status vs stroke
smokingStatusChart(stroke_data, unstroke_data)

# %%


def genderChart(stroke_data, unstroke_data):
    count_female_stroke = len(stroke_data[stroke_data['gender'] == '0'])
    count_male_stroke = len(stroke_data[stroke_data['gender'] == '1'])

    count_gender_stroke_map = {
        'female_stroke': count_female_stroke,
        'male_stroke': count_male_stroke
    }
    # generate data
    names = ['male', 'female']
    values = [count_male_stroke, count_female_stroke]

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Greens]
    colors = [a(0.4), b(0.4)]

    inner_ring = False
    figure_name = 'gender_stroke'
    title = 'Gender vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def ageChart(stroke_data, unstroke_data):
    count_age_stroke_map = {}
    for i in range(10):
        min_age = i*10
        max_age = (i+1)*10
        key = str(min_age) + '~' + str(max_age)
        count_age_stroke_map[key] = len(
            stroke_data[(stroke_data['age'] >= min_age) & (stroke_data['age'] < max_age)])

    # generate data
    names = []
    values = []
    colors = []
    a = plt.cm.Blues
    group_colors = []
    weight = 0.1
    for i in sorted(count_age_stroke_map.keys()):
        names.append(i)
        values.append(count_age_stroke_map[i])
        color = a(weight)
        colors.append(color)
        weight += 0.1

    figure_name = 'age_stroke'
    title = 'Age vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def hypertensionChart(stroke_data, unstroke_data):
    count_hypertension_stroke = len(
        stroke_data[stroke_data['hypertension'] == 1])
    count_unhypertension_stroke = len(
        stroke_data[stroke_data['hypertension'] == 0])

    count_hypertension_stroke_map = {
        'count_hypertension_stroke': count_hypertension_stroke,
        'count_unhypertension_stroke': count_unhypertension_stroke
    }

    # generate data
    names = ['hypertension', 'no hypertension']
    values = [count_hypertension_stroke, count_unhypertension_stroke]

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Greens]
    colors = [a(0.4), b(0.4)]

    figure_name = 'hypertension_stroke'
    title = 'Hypertension vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def heartDiseaseChart(stroke_data, unstroke_data):
    count_heart_disease_stroke = len(
        stroke_data[stroke_data['heart_disease'] == 1])
    count_not_heart_disease_stroke = len(
        stroke_data[stroke_data['heart_disease'] == 0])

    count_heart_disease_stroke_map = {
        'count_heart_disease_stroke': count_heart_disease_stroke,
        'count_not_heart_disease_stroke': count_not_heart_disease_stroke
    }

    # generate data
    names = ['heart disease', 'no heart disease']
    values = [count_heart_disease_stroke, count_not_heart_disease_stroke]

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Greens]
    colors = [a(0.4), b(0.4)]

    figure_name = 'heart_disease_stroke'
    title = 'Heart Disease vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def marriageChart(stroke_data, unstroke_data):
    count_married_stroke = len(stroke_data[stroke_data['ever_married'] == 1])
    count_unmarried_stroke = len(stroke_data[stroke_data['ever_married'] == 0])

    count_marriage_stroke_map = {
        'count_married_stroke': count_married_stroke,
        'count_unmarried_stroke': count_unmarried_stroke
    }

    # generate data
    names = ['married', 'not married']
    values = [count_married_stroke, count_unmarried_stroke]

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Greens]
    colors = [a(0.4), b(0.4)]

    figure_name = 'marriage_stroke'
    title = 'Marriage vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def workTypeChart(stroke_data, unstroke_data):
    count_children_stroke = len(
        stroke_data[stroke_data['work_type'] == 'children'])
    count_govt_job_stroke = len(
        stroke_data[stroke_data['work_type'] == 'govt_job'])
    count_private_stroke = len(
        stroke_data[stroke_data['work_type'] == 'private'])
    count_self_employed_stroke = len(
        stroke_data[stroke_data['work_type'] == 'self-employed'])

    count_work_type_stroke_map = {
        'count_children_stroke': count_children_stroke,
        'count_govt_job_stroke': count_govt_job_stroke,
        'count_private_stroke': count_private_stroke,
        'count_self_employed_stroke': count_self_employed_stroke
    }

    # generate data
    names = ['children', 'govt job', 'private', 'self-employed']
    values = [count_children_stroke, count_govt_job_stroke,
              count_private_stroke, count_self_employed_stroke]

    # Create colors
    a = plt.cm.Blues
    colors = [a(1.0), a(0.75), a(0.50), a(0.25)]

    figure_name = 'work_type_stroke'
    title = 'Work Type vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def residecTypeChart(stroke_data, unstroke_data):
    count_rural_stroke = len(
        stroke_data[stroke_data['residence_type'] == 'rural'])
    count_urban_stroke = len(
        stroke_data[stroke_data['residence_type'] == 'urban'])

    count_residence_type_stroke_map = {
        'count_rural_stroke': count_rural_stroke,
        'count_urban_stroke': count_urban_stroke
    }

    # generate data
    names = ['rural', 'urban']
    values = [count_rural_stroke, count_urban_stroke]

    # Create colors
    a, b = [plt.cm.Blues, plt.cm.Greens]
    colors = [a(0.4), b(0.4)]

    figure_name = 'residence_type_stroke'
    title = 'Residence Type vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def agvGlucoseChart(stroke_data, unstroke_data):
    count_glucose_level_stroke_map = {}
    for i in range(2, 12):
        min_glucose_level = i*25
        max_glucose_level = (i+1)*25
        key = str(min_glucose_level) + '~' + str(max_glucose_level)
        count_glucose_level_stroke_map[key] = len(
            stroke_data[(stroke_data['avg_glucose_level'] >= min_glucose_level) & (stroke_data['avg_glucose_level'] < max_glucose_level)])

    # generate data
    names = []
    values = []
    colors = []
    a = plt.cm.Blues
    group_colors = []
    weight = 0.1
    for i in range(2, 12):
        min_glucose_level = i*25
        max_glucose_level = (i+1)*25
        key = str(min_glucose_level) + '~' + str(max_glucose_level)
        names.append(key)
        values.append(count_glucose_level_stroke_map[key])
        color = a(weight)
        colors.append(color)
        weight += 0.1

    figure_name = 'avg_glucose_level_stroke'
    title = 'Avg Glucose Level vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def bmiChart(stroke_data, unstroke_data):
    count_bmi_stroke_map = {}
    for i in range(1, 10):
        min_bmi = i*10
        max_bmi = (i+1)*10
        key = str(min_bmi) + '~' + str(max_bmi)
        count_bmi_stroke_map[key] = len(
            stroke_data[(stroke_data['bmi'] >= min_bmi) & (stroke_data['bmi'] < max_bmi)])

    # generate data
    names = []
    values = []
    colors = []
    a = plt.cm.Blues
    group_colors = []
    weight = 0.1
    for i in range(1, 10):
        min_bmi = i*10
        max_bmi = (i+1)*10
        key = str(min_bmi) + '~' + str(max_bmi)
        names.append(key)
        values.append(count_bmi_stroke_map[key])
        color = a(weight)
        colors.append(color)
        weight += 0.1

    figure_name = 'bmi_stroke'
    title = 'Bmi vs Stroke'

    generateChart(names, values, colors, title, figure_name)


def smokingStatusChart(stroke_data, unstroke_data):
    count_smokes_stroke = len(
        stroke_data[stroke_data['smoking_status'] == 'smokes'])
    count_formerly_smoked_stroke = len(
        stroke_data[stroke_data['smoking_status'] == 'formerly smoked'])
    count_never_smoked_stroke = len(
        stroke_data[stroke_data['smoking_status'] == 'never smoked'])

    count_smoking_status_stroke_map = {
        'count_smokes_stroke': count_smokes_stroke,
        'count_formerly_smoked_stroke': count_formerly_smoked_stroke,
        'count_never_smoked_stroke': count_never_smoked_stroke
    }

    # generate data
    names = ['smokes', 'formerly smoked', 'never smoked']
    values = [count_smokes_stroke, count_formerly_smoked_stroke,
              count_never_smoked_stroke]

    # Create colors
    a = plt.cm.Blues
    colors = [a(0.33), a(0.66), a(0.99)]

    figure_name = 'smoking_status_stroke'
    title = 'Smoking Status vs Stroke'

    generateChart(names, values, colors, title, figure_name)


# %%
# generate and save pie chart
def generateChart(names, values, colors, title, figure_name):
    x = np.char.array(names)
    y = np.array(values)
    porcent = 100.*y/y.sum()
    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, porcent)]
    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(values, radius=1.2, colors=colors)
    plt.legend(mypie, labels, loc='left center', bbox_to_anchor=(-0.1, 1.),
               fontsize=10)
    plt.setp(mypie, width=0.4, edgecolor='white')

    ax.set_title(title)
    # save figure
    fig.savefig(os.path.join('../figures', figure_name), bbox_inches='tight')


# %%
