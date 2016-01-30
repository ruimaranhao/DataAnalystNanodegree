<div id="notebook" class="border-box-sizing" tabindex="-1">

<div id="notebook-container" class="container">

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

P5: Identifying Fraud from Enron Emails[¶](#P5:-Identifying-Fraud-from-Enron-Emails) {#P5:-Identifying-Fraud-from-Enron-Emails}
====================================================================================

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

In 2000, Enron was one of the largest companies in the United States. By
2002, it had collapsed into bankruptcy due to widespread corporate
fraud. In the resulting Federal investigation, a significant amount of
typically confidential information entered into the public record,
including tens of thousands of emails and detailed financial data for
top executives.

In this project, I use machine learning to identify persons of interest
based on financial and email data made public as a result of the Enron
scandal, as well as a labeled list of individuals who were indicted,
reached a settlement or plea deal with the government, or testified in
exchange for prosecution immunity.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Goals and dataset[¶](#Goals-and-dataset) {#Goals-and-dataset}
----------------------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

The goal of this project is to build a predictive model that can
identify persons of interest based on features included in the Enron
dataset. Such model could be used to find additional suspects who were
not indicted during the original investigation, or to find persons of
interest during fraud investigations at other businesses.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Infomartion regarding dataset[¶](#Infomartion-regarding-dataset) {#Infomartion-regarding-dataset}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[1\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    %matplotlib inline

    import sys
    import pickle
    import pprint
    import math
    sys.path.append("../tools/")

    from feature_format import featureFormat, targetFeatureSplit
    from tester import dump_classifier_and_data

    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[2\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[3\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    print "Number of employees: {}".format(len(data_dict))

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    Number of employees: 146

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[4\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # Employee names in this dataset
    for employee in data_dict:
        print employee

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    METTS MARK
    BAXTER JOHN C
    ELLIOTT STEVEN
    CORDES WILLIAM R
    HANNON KEVIN P
    MORDAUNT KRISTINA M
    MEYER ROCKFORD G
    MCMAHON JEFFREY
    HORTON STANLEY C
    PIPER GREGORY F
    HUMPHREY GENE E
    UMANOFF ADAM S
    BLACHMAN JEREMY M
    SUNDE MARTIN
    GIBBS DANA R
    LOWRY CHARLES P
    COLWELL WESLEY
    MULLER MARK S
    JACKSON CHARLENE R
    WESTFAHL RICHARD K
    WALTERS GARETH W
    WALLS JR ROBERT H
    KITCHEN LOUISE
    CHAN RONNIE
    BELFER ROBERT
    SHANKMAN JEFFREY A
    WODRASKA JOHN
    BERGSIEKER RICHARD P
    URQUHART JOHN A
    BIBI PHILIPPE A
    RIEKER PAULA H
    WHALEY DAVID A
    BECK SALLY W
    HAUG DAVID L
    ECHOLS JOHN B
    MENDELSOHN JOHN
    HICKERSON GARY J
    CLINE KENNETH W
    LEWIS RICHARD
    HAYES ROBERT E
    MCCARTY DANNY J
    KOPPER MICHAEL J
    LEFF DANIEL P
    LAVORATO JOHN J
    BERBERIAN DAVID
    DETMERING TIMOTHY J
    WAKEHAM JOHN
    POWERS WILLIAM
    GOLD JOSEPH
    BANNANTINE JAMES M
    DUNCAN JOHN H
    SHAPIRO RICHARD S
    SHERRIFF JOHN R
    SHELBY REX
    LEMAISTRE CHARLES
    DEFFNER JOSEPH M
    KISHKILL JOSEPH G
    WHALLEY LAWRENCE G
    MCCONNELL MICHAEL S
    PIRO JIM
    DELAINEY DAVID W
    SULLIVAN-SHAKLOVITZ COLLEEN
    WROBEL BRUCE
    LINDHOLM TOD A
    MEYER JEROME J
    LAY KENNETH L
    BUTTS ROBERT H
    OLSON CINDY K
    MCDONALD REBECCA
    CUMBERLAND MICHAEL S
    GAHN ROBERT S
    MCCLELLAN GEORGE
    HERMANN ROBERT J
    SCRIMSHAW MATTHEW
    GATHMANN WILLIAM D
    HAEDICKE MARK E
    BOWEN JR RAYMOND M
    GILLIS JOHN
    FITZGERALD JAY L
    MORAN MICHAEL P
    REDMOND BRIAN L
    BAZELIDES PHILIP J
    BELDEN TIMOTHY N
    DURAN WILLIAM D
    THORN TERENCE H
    FASTOW ANDREW S
    FOY JOE
    CALGER CHRISTOPHER F
    RICE KENNETH D
    KAMINSKI WINCENTY J
    LOCKHART EUGENE E
    COX DAVID
    OVERDYKE JR JERE C
    PEREIRA PAULO V. FERRAZ
    STABLER FRANK
    SKILLING JEFFREY K
    BLAKE JR. NORMAN P
    SHERRICK JEFFREY B
    PRENTICE JAMES
    GRAY RODNEY
    PICKERING MARK R
    THE TRAVEL AGENCY IN THE PARK
    NOLES JAMES L
    KEAN STEVEN J
    TOTAL
    FOWLER PEGGY
    WASAFF GEORGE
    WHITE JR THOMAS E
    CHRISTODOULOU DIOMEDES
    ALLEN PHILLIP K
    SHARP VICTORIA T
    JAEDICKE ROBERT
    WINOKUR JR. HERBERT S
    BROWN MICHAEL
    BADUM JAMES P
    HUGHES JAMES A
    REYNOLDS LAWRENCE
    DIMICHELE RICHARD G
    BHATNAGAR SANJAY
    CARTER REBECCA C
    BUCHANAN HAROLD G
    YEAP SOON
    MURRAY JULIA H
    GARLAND C KEVIN
    DODSON KEITH
    YEAGER F SCOTT
    HIRKO JOSEPH
    DIETRICH JANET R
    DERRICK JR. JAMES V
    FREVERT MARK A
    PAI LOU L
    BAY FRANKLIN R
    HAYSLETT RODERICK J
    FUGH JOHN L
    FALLON JAMES B
    KOENIG MARK E
    SAVAGE FRANK
    IZZO LAWRENCE L
    TILNEY ELIZABETH A
    MARTIN AMANDA K
    BUY RICHARD B
    GRAMM WENDY L
    CAUSEY RICHARD A
    TAYLOR MITCHELL S
    DONAHUE JR JEFFREY M
    GLISAN JR BEN F

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Strange two entries: TOTAL and THE TRAVEL AGENCY IN THE PARK. Let's
inspect them. By inspecting them, TOTAL is clearly an outlier and seems
to hold grand totals. The travel agency in the park is potentially not a
person.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[5\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    pprint.pprint(data_dict['THE TRAVEL AGENCY IN THE PARK'])
    pprint.pprint(data_dict['TOTAL'])

    y = []
    t = []
    for p in data_dict:
        t = t + [float(data_dict[p]['poi'])]
        y = y + [float(data_dict[p]['salary'])]

    x = range(0, len(data_dict))

    plt.scatter(x, y, c=t, cmap='jet')
    plt.title('Employee Salary (poi: False = blue; True = Red)')
    plt.xlabel('Employee')
    plt.ylabel('Salary')

    plt.show()

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 'NaN',
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 362096,
     'poi': False,
     'restricted_stock': 'NaN',
     'restricted_stock_deferred': 'NaN',
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 362096,
     'total_stock_value': 'NaN'}
    {'bonus': 97343619,
     'deferral_payments': 32083396,
     'deferred_income': -27992891,
     'director_fees': 1398517,
     'email_address': 'NaN',
     'exercised_stock_options': 311764000,
     'expenses': 5235198,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 83925000,
     'long_term_incentive': 48521928,
     'other': 42667589,
     'poi': False,
     'restricted_stock': 130322299,
     'restricted_stock_deferred': -7576788,
     'salary': 26704229,
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 309886585,
     'total_stock_value': 434509511}

</div>

</div>

<div class="output_area">

<div class="prompt">

</div>

<div class="output_png output_subarea">

![](data:image/png;base64,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%0AynH77bfnCasTJ05cFLIhISF89dVX11zThUFpGAbe3t55plssljzbw2q1Ov99rfvEP3Hhc3u+pvNy%0AcnLyrPvtt992DkbOnj17yWsULryG4J+oUqUKL7/8Mq+++io1atTgtttuwzAM+vXr53y95OTkOA+0%0Az9d0/jX4948Zzr/upOAp8M9as2bN8uyQw4cPZ+TIkcyZM4cGDRrw4YcfOqd9/PHHPPHEExe9Ecj1%0AuVQg/1N79uxh165dAMyfP586deoQHByMYRjOEdPBgwfZunUrAHv37mX9+vXce++9AHTq1IkVK1aw%0AY8cOHn74YQAefPBBli1bRmJiIgALFixwfgugQYMGfPbZZ+Tk5GC32xk5ciRvvvnmRXV5eXkxadIk%0AEhISnPcdOHCA0qVLU6hQIdauXUuvXr1o27YtRYoU4bfffsNut+fZNr/99htNmjShS5cuVK9enRUr%0AVjjnudS269atGxMnTqRGjRqEhYVdNL1MmTIkJSXlCY3Lbb/69evzv//9D3C8qS9YsIAHH3zwmp6T%0Aq1m7di2PPvooHTt2pHz58qxcufKi3levXk2vXr2oVasWzzzzDO3bt2f37t3UqFGD+Ph44uLiANi1%0AaxctWrRwPlf/hmEYLFu2jJycHLKzs/nyyy9p2LBhnnmKFCnC3r17ycnJITc3l1WrVjmnXes+8U/q%0A+fu6t2/fDjiuh/j111+d0+rXr8/s2bMxDIOcnByefvpp5s6d+6/XfSmtWrWiVq1avP766851Lliw%0AgLS0NADeffddoqOjKVy4MNWqVWPhwoUA7Ny5k507d+ZZ1uHDh53X3EjBUuBH+n+3f/9+YmJiAMeF%0AK+cviLLb7axevZohQ4bkX3FuqmfPnheNBAYPHoy/v/8VH3fhSKZIkSK88847HD58mKJFizJhwoQ8%0A84SGhvL2228zZswYMjMz8fDwYPz48ZQrV875+Lvuuovbb7/dWUv9+vXp27cvvXv3xmKxEBwczHvv%0AvQfAU089xYQJE3j00Uex2+1UrVqV6Ojoi2p89NFHyczMpH///uTk5GCxWKhYsSIzZ87Ew8ODp59+%0AmokTJzJjxgyKFClC8+bNiY+Pz9Njly5deOGFF2jfvj0hISE89NBDfPzxx84Dmr+P6Bo3bsywYcMu%0Ae1o6JCSEOnXqEBsb6wy1y22/YcOG8dprr9GmTRtycnJo2LCh87T4heu93IV8V7oivnfv3owYMYIl%0AS5YQGhpK06ZNWbNmTZ7HNmzYkDVr1tC6dWsCAgIoXLgwr732mrPeSZMmkZ2djd1uZ9KkSXlO7f9T%0AFouFMmXK0K1bNzIyMnj44Yedp6jP91G/fn3q1q1LixYtKF68OPfdd5/zIscr7RP/5kK+vz+3bdu2%0AZc2aNTRr1ozw8PA8H+sMGzaMsWPH0rZtW6xWKw8++CB9+/b919viwp4vNHz4cNq2bcvatWvp1KkT%0AJ06c4LHHHsNisVCqVCnGjx8PwBtvvMHLL7/M559/7jwrc6G1a9fy9ttvX1d9kk9ceQHBzXL48GGj%0Ac+fOhmEYRlRUlHHs2DHDMAxj3bp1xvLlyw3DMIydO3caAwYMyLca5fJiY2ONFi1aXNcyTp8+bURG%0ARhrHjx+/QVXlnw0bNhitW7e+4jwbN240+vXrZxjGjdl+cmVpaWnG888/n99l3BJiY2PzXJgqBYvL%0AR/o2m41hw4Zx6NAhLBYLo0aN4o477nBOX7lyJdOmTcPLy4uOHTvSqVOna1ru+aPamJgYXnzxRWw2%0AGxaLxXkq69ChQ5QtW/bGNyQ3xPV8x3rBggW8+eabDBgwgPDw8BtYlesNHTqUuLg4Jk6ceMX5atWq%0ARYUKFfjll1/w9fXN198yMIOdO3fy3HPP5XcZ+c5ms/HRRx8531el4LEYxg34QPYfWLFiBatWrWLs%0A2LGsW7eO2bNnM23aNMBxUU2rVq1YvHgxfn5+dO3alRkzZlzXFbQiIiLi4PIL+Zo2bcro0aMBOHr0%0AKIUKFXJO279/v/NrXd7e3tSpU8d5oY+IiIhcn3y5kM/T05Po6Gh+/PHHPD/lmJaWRnBwsPN2YGAg%0Aqamp+VGiiIiI28m3r+yNHz+eH374geHDh5OVlQU4vtN74c89pqen5zkTcCku/nRCRESkwHL5SH/J%0AkiWcOHGC/v37O/8a1fmLkCpWrEh8fDwpKSn4+/sTFxdHnz59rrg8i8VCYmLBPxsQFhZc4Ptwhx7A%0APfpwhx5AfdxK3KEHcI8+wsKCrz7TZbg89Fu0aEF0dDTdu3cnNzeXV199lR9//JGMjAw6d+5MdHQ0%0Affr0wW63ExUVRfHixV1dooiIiFtyeej7+fnx1ltvXXZ6kyZNrvh3ukVEROTfKfA/wysiIiLXRqEv%0AIiJiEgp9ERERk1Doi4iImIRCX0RExCQU+iIiIiah0BcRETEJhb6IiIhJKPRFRERMQqEvIiJiEgp9%0AERERk1Doi4iImIRCX0RExCQU+iIiIiah0BcRETEJhb6IiIhJKPRFRERMQqEvIiJiEgp9ERERk1Do%0Ai4iImIRCX0RExCQU+iIiIibhld8FiIiY3fz5y9iwIZ6yZQvx1FNd8fDQeExuDoW+iEg+mj59PmPH%0A7iI7OxhI4dCht5g8+b/5XZa4KR1Oiojko1Wr9p8LfABffv31RL7WI+5NoS8iko8CA/O+DQcFeeZT%0AJWIGCn0RkXw0ZMijVK16EovlFGXKHGfw4Gb5XZK4MX2mLyKSj6pVi+D772P4888/KVmyBMHBIfld%0Akrgxhb6ISD7z8/PjzjvvzO8yxAR0el9ERMQkFPoiIiImodAXERExCYW+iIiISSj0RURETEKhLyIi%0AYhIKfREREZNw+ff0rVYrr7zyCgkJCeTk5DBw4EAiIyOd02fPns2iRYsIDQ0FYPTo0VSoUMHVZYqI%0AiLgdl4f+N998Q5EiRZg0aRIpKSm0b98+T+hv376diRMnUrVqVVeXJiIi4tZcHvotWrSgefPmANjt%0Adjw98/5xie3btzN9+nROnTpF48aN6devn6tLFBERcUsuD/2AgAAA0tLSGDRoEIMHD84zvVWrVjz+%0A+OMEBgbyzDPPsHr1aho3buzqMkVERNyOxTAMw9UrPXbsGM888wyPP/44HTp0yDMtLS2NoKAgAObO%0AnUtycjJPPfWUq0sUERFxOy4f6Z86dYrevXszcuRI6tWrl2daamoqbdu2ZdmyZfj7+xMbG0tUVNRV%0Al5mYmHqzynWZsLDgAt+HO/QA7tGHO/QA6uNW4g49gHv0ERYW/K8f6/LQnz59Oqmpqbz33nu89957%0AAHTu3JnMzEw6d+7MkCFD6NmzJz4+PjzwwAM0bNjQ1SWKiIi4pXw5vX+jFfSjNnCfo8+C3gO4Rx/u%0A0AOoj1uJO/QA7tHH9Yz09eM8IiIiJqHQFxERMQmFvoiIiEko9EVERExCoS8iImISCn0RERGTUOiL%0AiIiYhEJfRETEJBT6IiIiJqHQFxERMQmFvoiIiEko9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJf%0ARETEJBT6IiIiJqHQFxERMQmFvoiIiEko9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6%0AIiIiJqHQFxERMQmFvoiIiEko9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6IiIiJqHQ%0AFxERMQmFvoiIiEl4uXqFVquVV155hYSEBHJychg4cCCRkZHO6StXrmTatGl4eXnRsWNHOnXq5OoS%0ARURE3JLLQ/+bb76hSJEiTJo0iZSUFNq3b+8MfavVyvjx41m8eDF+fn507dqVyMhIihYt6uoyRURE%0A3I7LT++3aNGC5557DgC73Y6np6dz2v79+ylbtizBwcF4e3tTp04d4uLiXF2iiIiIW3L5SD8gIACA%0AtLQ0Bg0axODBg53T0tLSCA4Odt4ODAwkNTXV1SWKiIi4pXy5kO/YsWP06tWL9u3b06pVK+f9wcHB%0ApKenO2+np6dTqFCh/ChRRETE7VgMwzBcucJTp07Ro0cPRo4cSb169fJMs1qttG7dmgULFuDv70+X%0ALl2YPn06xYsXd2WJIiIibsnloT9mzBi+//57KlSo4Lyvc+fOZGZm0rlzZ1atWsV7772H3W4nKiqK%0Abt26XXWZiYkF/yOAsLDgAt+HO/QA7tGHO/QA6uNW4g49gHv0ERYWfPWZLsPloX8zFPQnENxnRyzo%0APYB79OEOPYD6uJW4Qw/gHn1cT+jrx3lERERMQqEvIiJiEgp9ERERk1Doi4iImIRCX0RExCQU+iIi%0AIiah0BcRETEJhb6IiIhJKPRFRERMQqEvIiJiEgp9ERERk1Doi4iImIRCX0RExCQU+iIiIiah0BcR%0AETEJhb6IiIhJKPRFRERMQqEvIiJiEgp9ERERk1Doi4iImIRCX0RExCQU+iIiIiah0BcRETEJhb6I%0AiIhJKPRFRERMQqEvIiJiEgp9ERERk1Doi4iImIRCX0RExCQU+iIiIiah0BcRETEJhb6IiIhJKPRF%0ARERMQqEvIiJiEgp9ERERk1Doi4iImES+hf6WLVvo0aPHRffPnj2b1q1b06NHD3r06MHBgwfzoToR%0AERH345UfK/3www/5+uuvCQwMvGja9u3bmThxIlWrVs2HykRERNxXvoz0y5Urx7vvvothGBdN2759%0AO9OnT6dbt2588MEH+VCdiIiIe8qX0G/WrBmenp6XnNaqVStGjx7NJ598woYNG1i9erVrixMREXFT%0A+XJ6/0p69epFUFAQAI0aNWLHjh00btz4io8JCwt2QWU3nzv04Q49gHv04Q49gPq4lbhDD+A+ffwb%0At1Top6am0rZtW5YtW4a/vz+xsbFERUVd9XGJiakuqO7mCgsLLvB9uEMP4B59uEMPoD5uJe7QA7hH%0AH9dz0JKvoW+xWABYunQpGRkZdO7cmSFDhtCzZ098fHx44IEHaNiwYX6WKCIi4jYsxqWupitgCvpR%0AG7jP0WdB7wHcow936AHUx63EHXoA9+jjekb6+nEeERERk1Doi4iImIRCX0RExCQU+iIiIiZx1dBP%0ATEx0RR0iIiJyk1019B9//HH69evHd999h9VqdUVNIiIichNcNfR/+OEHnnzySX755ReaN2/OqFGj%0A2LZtmytqExERkRvoqj/OY7FYqFu3LnfddRffffcdb775JqtWraJIkSIMHz6cWrVquaJOERERuU5X%0ADf21a9fy9ddfs3btWho1asRbb71F7dq12b17N3379uWXX35xRZ0iIiJyna4a+tOmTaNjx46MHDmS%0AgIAA5/0RERH06dPnphYnIiIiN85VQ9/Hx4cOHTpcctp//vOfG12PiIiI3CRXvZAvOzubhIQEV9Qi%0AIiIiN9FVR/pJSUlERkZStGhRfH19AcfFfT/99NNNL05ERERunKuG/syZMy+67/yfxBUREZGC46qh%0AHxYWxs8//0xGRgYANpuNI0eOMGjQoJtenIiIiNw4Vw39Z555hqysLOLj46lbty5xcXE89NBDrqhN%0AREREbqCrXsh38OBBPv30Ux5++GH69OnDwoULOXbsmCtqExERkRvoqqFfrFgxLBYLFStWZPfu3YSH%0Ah+uP8IiIiBRAVz29X6lSJV577TW6dOnCiy++yMmTJ8nJyXFFbSIiInIDXXWkP2rUKFq2bMkdd9zB%0As88+S2JiIlOmTHFFbSIiInIDXXakv27dOudX8wzDIC4ujuDgYJo1a0ZKSorLChQREZEb47KhP3Xq%0A1Cs+cM6cOTe8GBEREbl5Lhv6CnURERH3ctUL+davX8/MmTPJzMzEbrdjt9s5duwYK1eudEV9IiIi%0AcoNc9UK+V199laZNm2Kz2ejevTvlypWjV69erqhNREREbqCrhr6fnx9RUVHUrVuXkJAQxowZww8/%0A/OCK2kREROQGuqbQT05OpkKFCmzZsgWLxUJSUpIrahMREZEb6Kqh/5///Ifnn3+eyMhIlixZQuvW%0AralWrZorahMREZEb6IoX8q1cuZJq1aoxa9YsfvrpJ8LDw/H19WXcuHGuqk9ERERukMuO9D/66CPe%0AffddcnJy2LNnDy+88AKtW7emfPnyTJo0yZU1ioiIyA1w2ZH+kiVLmD9/PgEBAUyePJmHHnqITp06%0AYRgGLVu2dGWNIiIicgNcdqTv4eFBQEAAAL///jv169cHwGKxOH+eV0RERAqOy470PT09SUlJITMz%0Ak507dzpDPyEhAS+vq/6mj4iIiNxiLpve/fr149FHH8VqtRIVFUXx4sX57rvveOONN3j66addWaOI%0AiIjcAJcN/RYtWlCrVi3OnDlD5cqVAfD392fMmDHcd999LitQREREbowrnqcPDw8nPDzcebtx48Y3%0Aux4RERG5Sa764zw3y5YtW+jRo8dF969cuZKoqCi6dOnCwoUL86EyERER95QvV+R9+OGHfP311wQG%0ABua532q1Mn78eBYvXoyfnx9du3YlMjKSokWL5keZIiIibiVfRvrlypXj3XffxTCMPPfv37+fsmXL%0AEhwcjLe3N3Xq1CEuLi4/ShQREXE7+RL6zZo1w9PT86L709LSCA4Odt4ODAwkNTXVlaWJiIi4rVvq%0AC/fBwcGkp6c7b6enp1OoUKGrPi4sLPiq8xQE7tCHO/QA7tGHO/QA6uNW4g49gPv08W/cUqFfsWJF%0A4uPjSUlJwd/fn7i4OPr06XPVxyUmFvyzAWFhwQW+D3foAdyjD3foAdTHrcQdegD36ON6DlryNfTP%0A/5zv0qVLycjIoHPnzkRHR9OnTx/sdrvzR4FERETk+uVb6JcuXZp58+YB0Lp1a+f9TZo0oUmTJvlV%0AloiIiNvKt+/pi4iIiGsp9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6IiIiJqHQFxER%0AMQmFvoiIiEko9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6IiIiJqHQFxERMQmFvoiI%0AiEko9EVERExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6IiIiJqHQFxERMQmFvoiIiEko9EVE%0ARExCoS8iImISCn0RERGTUOiLiIiYhEJfRETEJBT6IiIiJqHQFxERMQmFvoiIiEko9EVERExCoS8i%0AImISCn0RERGT8HL1Cu12OzExMezZswdvb2/Gjh1L2bJlndNnz57NokWLCA0NBWD06NFUqFDB1WWK%0AiIi4HZeH/ooVK7BarcybN48tW7Ywfvx4pk2b5py+fft2Jk6cSNWqVV1dmoiIiFtzeehv3LiRBg0a%0AAFCjRg3++OOPPNO3b9/O9OnTOXXqFI0bN6Zfv36uLlFERMQtufwz/bS0NIKCgpy3PT09sdvtztut%0A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%0A4K677mLHjh3cc889hIaGsn37dnbv3o2TkxN6vZ6oqKiRKk8IIYS4pY1o0xdCCCGE/cjFeYQQQggH%0AIeiuYFkAAApNSURBVE1fCCGEcBDS9IUQQggHocmmbzKZSEpKwmAwEBMTQ2NjIwCNjY1ERUURGxtL%0AUVGRnav8Z1arldzcXGJiYjAYDLS1tdm7pOvW19dHWloa8fHxREZGUl1dTWtrK7GxscTHx7Nu3Tq0%0AdLpIV1cX8+bN4/Tp05rMsXXrVmJiYoiIiODTTz/VZAar1UpWVpat7lOnTmkqx08//YTBYAD4y7or%0AKiqIiIggOjqab7/91o7V/rWhOY4dO0Z8fDwGg4Hly5fT1dUFjP4cQzMMMhqNxMTE2G6P9gwwPEdX%0AVxerVq0iISGB+Ph4zp49C9xEDqVBW7ZsUTt37lRKKXXq1Cml1+uVUko99dRTqq2tTSml1IoVK9TR%0Ao0ftVuP12Lt3r8rMzFRKKdXY2KhWrVpl54quX2VlpVq/fr1SSqk//vhDzZs3TyUlJan6+nqllFK5%0Aubnqq6++smeJ1623t1etXr1aPfnkk+rkyZNq5cqVmspRV1enVq5cqZRSqru7W73zzjua3Bffffed%0ASklJUUopVVNTo5KTkzWTY9u2bSo8PFxFR0crpdQ1n0MdHR0qPDxc9fb2KpPJpMLDw1VPT489y77K%0AlTkSEhLUsWPHlFJKffzxxyo/P191dnaO6hxXZlBKqSNHjqhnnnnGtkyL+yIjI0Pt2bNHKTXwmq+u%0Arr6pHJoc6ScmJhIdHQ2AxWLBxcUFs9lMX18fnp6eAAQHB1NbW2vPMv/RgQMHeOSRRwB48MEHaWpq%0AsnNF1y8sLIwXX3wRGBihOTk5cfToUQICAgAICQkZ9dt/0IYNG4iNjcXDwwNAczlqamqYOXMmq1ev%0AJikpidDQUI4cOaKpDAC33XYbJpMJpRQmkwlnZ2fN5PDy8qKoqMg2or/Wc+jw4cP4+fnh7OyMm5sb%0AXl5eNDc327Psq1yZo7CwkFmzZgH/e689dOjQqM5xZYYLFy6wefNmsrOzbctGewa4OsfBgwdpb2/n%0A2WefxWg0EhQUdFM5Rn3T37VrFwsXLhz209raiouLC52dnaSnp5OamorZbB42G9+1LvM72lxZ85gx%0AY7BarXas6Pq5urpyxx13YDabSUlJ4aWXXhpWu6ur66jf/gBVVVXceeedBAcHA6CUGnYIWQs5zp8/%0A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%0A)

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[6\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    ## Are there duplicates?
    empl2set = set(data_dict.keys())
    if len(empl2set) != len(data_dict):
        print "WARNING: DUPLICATES FOUND!"
    else:
        print "NO DUPLICATES FOUND!"

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    NO DUPLICATES FOUND!

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[7\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    print "Number of features: {}".format(len(data_dict['TOTAL'].keys()))
    pprint.pprint(data_dict['TOTAL'].keys())

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    Number of features: 21
    ['salary',
     'to_messages',
     'deferral_payments',
     'total_payments',
     'exercised_stock_options',
     'bonus',
     'restricted_stock',
     'shared_receipt_with_poi',
     'restricted_stock_deferred',
     'total_stock_value',
     'expenses',
     'loan_advances',
     'from_messages',
     'other',
     'from_this_person_to_poi',
     'poi',
     'director_fees',
     'deferred_income',
     'long_term_incentive',
     'email_address',
     'from_poi_to_this_person']

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[8\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    #Let's see how many values are NaN
    def NaN_counter(feature_name):
        "Calculates the percentage of NaNs in a feature"
        count_NaN = 0
        for employee in data_dict:
            if math.isnan(float(data_dict[employee][feature_name])):
                count_NaN += 1
        percent_NaN = 100*float(count_NaN)/float(len(data_dict))
        percent_NaN = round(percent_NaN,2)
        return percent_NaN

    print str(NaN_counter('salary'))
    print str(NaN_counter('exercised_stock_options'))
    print str(NaN_counter('bonus'))

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    34.93
    30.14
    43.84

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[9\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    #find number of persons of interest
    poi = 0
    for p in data_dict:
        if data_dict[p]['poi']:
            poi += 1
    print poi

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    18

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Summary[¶](#Summary) {#Summary}

The dataset contains a total of 146 data points, each with 21 features.
Of the 146 records, 18 are labeled as persons of interest. Two of these
entries are to be removed because they are not persons.

Furthermore, there are some high percentages of NaN. As an example,
34.72% of the salaries, 29.86% of exercised\_stock\_options, and 43.75%
of bonus are NaN.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Task 1: Select what features will be used.[¶](#Task-1:-Select-what-features-will-be-used.) {#Task-1:-Select-what-features-will-be-used.}
------------------------------------------------------------------------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[10\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # this list is augmented after Task 3.
    features_list = ['poi', 'salary']

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Task 2: Remove outliers.[¶](#Task-2:-Remove-outliers.) {#Task-2:-Remove-outliers.}
------------------------------------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[11\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
    data_dict.pop('TOTAL', 0)

    print "\nNumber of employees: {}". format(len(data_dict) - 2)

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    Number of employees: 142

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Task 3: Create new feature(s)[¶](#Task-3:-Create-new-feature(s)) {#Task-3:-Create-new-feature(s)}
----------------------------------------------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[12\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    #feature: wealth - salary, total stock value, exercised stock option, bonus.
    for employee in data_dict:
        wealth = 0
        if not math.isnan(float(data_dict[employee]['exercised_stock_options'])):
            wealth += float(data_dict[employee]['exercised_stock_options'])
        if not math.isnan(float(data_dict[employee]['salary'])):
            wealth += float(data_dict[employee]['salary'])
        if not math.isnan(float(data_dict[employee]['bonus'])):
            wealth += float(data_dict[employee]['bonus'])
        if not math.isnan(float(data_dict[employee]['total_stock_value'])):
            wealth += float(data_dict[employee]['total_stock_value'])
        data_dict[employee]['wealth'] = wealth

        fPOI = 0
        sPOI = 0
        if not math.isnan(float(data_dict[employee]['from_poi_to_this_person'])):
            fPOI = float(data_dict[employee]['from_poi_to_this_person'])
        if not math.isnan(float(data_dict[employee]['from_this_person_to_poi'])):
            sPOI = float(data_dict[employee]['from_this_person_to_poi'])

        if fPOI + sPOI == 0:
            data_dict[employee]['ratio_sent_poi'] = 0
            data_dict[employee]['ratio_rcv_poi'] = 0
        else:
            data_dict[employee]['ratio_sent_poi'] = sPOI / (sPOI + fPOI)
            data_dict[employee]['ratio_rcv_poi'] = fPOI / (sPOI + fPOI)

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

I created three features:

-   fraction\_from\_poi: Fraction of emails received from POIs.

-   fraction\_to\_poi: Fraction of emails sent to POIs.

-   wealth: Salary, total stock value, exercised stock options
    and bonuses.

Non of these features seemed to affect the performance of the algorithm
using the selected features in the dataset.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### After removing outliers and add new features to the dataset, I am using SelectKBest to find the best features. This has been used to update the feature\_list defined before.[¶](#After-removing-outliers-and-add-new-features-to-the-dataset,-I-am-using-SelectKBest-to-find-the-best-features.-This-has-been-used-to-update-the-feature_list-defined-before.) {#After-removing-outliers-and-add-new-features-to-the-dataset,-I-am-using-SelectKBest-to-find-the-best-features.-This-has-been-used-to-update-the-feature_list-defined-before.}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[13\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    flist = ['salary',
     'to_messages',
     'deferral_payments',
     'total_payments',
     'exercised_stock_options',
     'bonus',
     'restricted_stock',
     'shared_receipt_with_poi',
     'restricted_stock_deferred',
     'total_stock_value',
     'expenses',
     'loan_advances',
     'from_messages',
     'other',
     'from_this_person_to_poi',
     'poi',
     'director_fees',
     'deferred_income',
     'long_term_incentive',
     'from_poi_to_this_person']

    data = featureFormat(my_dataset, flist, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    kbest = SelectKBest(f_regression, k=5)

    X_new = kbest.fit_transform(features, labels)

    pairs = sorted(zip(flist, kbest.scores_), key=lambda x: x[1], reverse=True)

    pprint.pprint(pairs)

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    [('exercised_stock_options', 102.45258245307906),
     ('deferred_income', 64.196870594214801),
     ('from_messages', 59.509832726244511),
     ('bonus', 57.412332328908263),
     ('restricted_stock_deferred', 52.526260661330944),
     ('deferral_payments', 49.913259030355306),
     ('restricted_stock', 49.551893864045951),
     ('total_payments', 35.552051858302711),
     ('long_term_incentive', 27.754547277447401),
     ('salary', 25.995931831498432),
     ('expenses', 25.055852430123178),
     ('total_stock_value', 20.7467993205582),
     ('from_this_person_to_poi', 18.289684043404524),
     ('director_fees', 17.145051599663137),
     ('poi', 14.499919583848534),
     ('to_messages', 8.8086536471804351),
     ('other', 6.2248683163393332),
     ('loan_advances', 2.9376171731188969),
     ('shared_receipt_with_poi', 1.1304454953978316)]

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

#### Extending feature\_list[¶](#Extending-feature_list) {#Extending-feature_list}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[14\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    # this list is augmented after Task 3.
    features_list = features_list + ['bonus', 
                                     'total_stock_value',
                                     'exercised_stock_options']

    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Task 4: Try a varity of classifiers[¶](#Task-4:-Try-a-varity-of-classifiers) {#Task-4:-Try-a-varity-of-classifiers}
----------------------------------------------------------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

#### GaussianNB[¶](#GaussianNB) {#GaussianNB}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[15\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # Provided to give you a starting point. Try a variety of classifiers.

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    dump_classifier_and_data(clf, my_dataset, features_list)

    import tester
    tester.main()

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    GaussianNB()
        Accuracy: 0.84677   Precision: 0.50312  Recall: 0.32300 F1: 0.39342 F2: 0.34791
        Total predictions: 13000    True positives:  646    False positives:  638   False negatives: 1354   True negatives: 10362

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

#### KNeighborsClassifier[¶](#KNeighborsClassifier) {#KNeighborsClassifier}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[16\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=5, p=2, weights='distance')

    dump_classifier_and_data(clf, my_dataset, features_list)

    import tester
    tester.main()

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=5, p=2, weights='distance')
        Accuracy: 0.87800   Precision: 0.69602  Recall: 0.36750 F1: 0.48102 F2: 0.40581
        Total predictions: 13000    True positives:  735    False positives:  321   False negatives: 1265   True negatives: 10679

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

#### DecisionTreeClassifier[¶](#DecisionTreeClassifier) {#DecisionTreeClassifier}

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[17\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    from sklearn import tree
    from sklearn.metrics import accuracy_score
    clf = tree.DecisionTreeClassifier(min_samples_split=40)

    dump_classifier_and_data(clf, my_dataset, features_list)

    import tester
    tester.main()

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
                min_samples_split=40, min_weight_fraction_leaf=0.0,
                random_state=None, splitter='best')
        Accuracy: 0.82385   Precision: 0.26984  Recall: 0.08500 F1: 0.12928 F2: 0.09849
        Total predictions: 13000    True positives:  170    False positives:  460   False negatives: 1830   True negatives: 10540

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Task 5: Tune your classifier[¶](#Task-5:-Tune-your-classifier) {#Task-5:-Tune-your-classifier}
--------------------------------------------------------------

Tune your classifier to achieve better than .3 precision and recall
using our testing script. Check the tester.py script in the final
project folder for details on the evaluation method, especially the
test\_classifier function. Because of the small size of the dataset, the
script uses stratified shuffle split cross validation. For more info:
<http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

I have selected the KNeighborsClassifier(algorithm='auto',
leaf\_size=30, metric='minkowski', metric\_params=None, n\_neighbors=5,
p=2, weights='distance') classifier as the best performer because it
shows better recall (i.e. the proportion of persons identified as POIs,
who actually are POIs) than the alternatives at the expense of
precision. I believe however that recall is more important in this
context. False positives can always be double-checked manually.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[18\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', 
                               metric_params=None, n_neighbors=5, p=2, weights='distance')

    from sklearn.cross_validation import train_test_split

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.5, random_state=42)
        
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)


    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)

    dump_classifier_and_data(clf, my_dataset, features_list)

    import tester
    tester.main()

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_neighbors=5, p=2, weights='distance')
        Accuracy: 0.87800   Precision: 0.69602  Recall: 0.36750 F1: 0.48102 F2: 0.40581
        Total predictions: 13000    True positives:  735    False positives:  321   False negatives: 1265   True negatives: 10679

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

The StratifiedShuffleSplit does not seem to have an impact in the
classifier.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Validation allows to evaluate the performance of a algorithm. It gives
us more evidence to draw conclusions wrt. generalization beyond the
dataset used to train it (overfitting). One of the biggest mistakes one
can make is to use the same data fro training and testing.

To cross validate algorithm I thought was best, I ran 1000 randomized
trials and evaluated the mean evaluation metrics. Given the imbalance in
the dataset betweet POIs and non-POIs, accuracy would not have been an
appropriate evaluation metric. I used precision and recall instead:

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[29\]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from numpy import mean
    import progressbar

    precision, recall = [], []
    progress = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                progressbar.Percentage(), ' ',
                                                progressbar.ETA()])
    for it in progress(range(1000)):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=it)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        precision = precision + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
            
    print '\nPrecision:', mean(precision)
    print 'Recall:', mean(recall)

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stderr output_text">

</div>

</div>

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stdout output_text">

    Precision: 0.583985714286
    Recall: 0.30011468254

</div>

</div>

<div class="output_area">

<div class="prompt">

</div>

<div class="output_subarea output_stream output_stderr output_text">

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Conclusions[¶](#Conclusions) {#Conclusions}
----------------------------

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="prompt input_prompt">

</div>

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

I wasn't able to find other features that would improve the classifier.
I think the next steps in searching for a better performing classifier
would be to use MinMaxScaler to have all variables within the same range
and GridSearchCV parameter optimization. The latter can however be
specific for the current data, and may not generalize.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">

In \[ \]:

</div>

<div class="inner_cell">

<div class="input_area">

<div class="highlight hl-ipython2">

     

</div>

</div>

</div>

</div>

</div>

</div>

</div>
