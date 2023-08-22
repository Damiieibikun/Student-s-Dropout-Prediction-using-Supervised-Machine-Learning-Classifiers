# Prediction Analysis on Student's Dropout
A Data science project on Predicting Student's dropout using Machine Learning classification models

### Introduction ###
The goal of this project is to develop a predictive model using machine learning classification algorithms to identify students who are likely to drop out. By leveraging data on student demographics, academic performance, socio-economic factors, and other relevant variables, we aim to build a robust predictive model that can effectively forecast the likelihood of student dropping out. This model can then be used by educational institutions to allocate resources, implement early intervention strategies, and support at-risk students.

### Problem ###
In today's educational landscape, student retention and success are of utmost importance for educational institutions. Identifying students who are at risk of dropping out and implementing timely interventions can greatly contribute to improving graduation rates and ensuring academic success. 
What are certain factors that may affect student's retention or dropout in academic institutions?

### Methodology Approach ###
The methodology adopted during the course of this project include
* Data collected through [Kaggle datasets](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention). 
* Data processing and descriptive analysis
* Exploratory Data Analysis using Python visualization tools to gain insights into the data and identify any patterns or trends.
* Predictive analysis using Machine Learning Classification algorithms.

### Dependencies ###
* Data Wrangling and processing Libraries
    * Pandas
    * Numpy
* Visualization Libraries
    * Matplotlib
    * Seaborn
* Machine Learning Libraries
    * Scikit Learn

### Deliverables ###
* Comprehensive Jupyter notebooks containing codes and results

### Findings/Results ###
Through analysis, it is deduced that several factors are to be considered when predicting the students retention/dropout (not limited to):
* Age at enrollment
* Tuition fees up to date
* Debt
* Grades for Curricular units 1st and 2nd semester
* Approved units for 1st and 2nd semester
* Scholarship holders
* Parents's occupation

Out of five classification algorithms, the ones that performed best after several tries include:
* Support Vector Classifiers
* Random Forest
* Logistic Regression


# Appendix

The table below gives a brief description for each column/field in the dataset

| S/N | Field | Description | Categories explained |
| ----------- | ----------- | ----------- |----------- |
|1.| Marital status | The marital status of the student.| 1—Single<br>2—Married<br>3—Widower<br>4—Divorced<br>5—Facto union<br> 6—Legally separated|
|2.| Application mode | Method of application used by student| 1—1st phase—general contingent <br> 2—Ordinance No. 612/93 <br> 3—1st phase—special contingent (Azores Island) <br> 4—Holders of other higher courses <br> 5—Ordinance No. 854-B/99 <br> 6—International student (bachelor) <br> 7—1st phase—special contingent (Madeira Island) <br> 8—2nd phase—general contingent <br> 9—3rd phase—general contingent <br> 10—Ordinance No. 533-A/99, item b2) (Different Plan) <br> 11—Ordinance No. 533-A/99, item b3 (Other Institution) <br> 12—Over 23 years old <br> 13—Transfer <br> 14—Change in course <br> 15—Technological specialization diploma holders <br> 16—Change in institution/course <br> 17—Short cycle diploma holders <br> 18—Change in institution/course (International)|
|3.| Application order | The order in which the student applied | |
|4.| Course | The course taken by the student | 1—Biofuel Production Technologies <br> 2—Animation and Multimedia Design <br> 3—Social Service (evening attendance) <br> 4—Agronomy <br> 5—Communication Design <br> 6—Veterinary Nursing <br> 7—Informatics Engineering <br> 8—Equiniculture <br> 9—Management <br> 10—Social Service <br> 11—Tourism <br> 12—Nursing <br> 13—Oral Hygiene <br> 14—Advertising and Marketing Management <br> 15—Journalism and Communication <br> 16—Basic Education <br> 17—Management (evening attendance)|
|5.| Daytime/evening attendance | Whether the student attends classes during the day or in the evening | 1—daytime <br> 0—evening|
|6.| Previous qualification | The qualification obtained by the student before enrolling in higher education | 1—Secondary education <br> 2—Higher education—bachelor’s degree <br> 3—Higher education—degree<br>4—Higher education—master’s degree<br>5—Higher education—doctorate<br>6—Frequency of higher education<br>7—12th year of schooling—not completed<br>8—11th year of schooling—not completed<br>9—Other—11th year of schooling<br>10—10th year of schooling<br>11—10th year of schooling—not completed<br>12—Basic education 3rd cycle (9th/10th/11th year) or equivalent<br>13—Basic education 2nd cycle (6th/7th/8th year) or equivalent<br>14—Technological specialization course<br>15—Higher education—degree (1st cycle)<br>16—Professional higher technical course<br>17—Higher education—master’s degree (2nd cycle) |
|7.| Nationality | The nationality of the student | 1—Portuguese<br>2—German<br>3—Spanish<br>4—Italian<br>5—Dutch<br>6—English<br>7—Lithuanian<br>8—Angolan<br>9—Cape Verdean<br>10—Guinean<br>11—Mozambican<br>12—Santomean<br>13—Turkish<br>14—Brazilian<br>15—Romanian<br>16—Moldova (Republic of)<br>17—Mexican<br>18—Ukrainian<br>19—Russian<br>20—Cuban<br>21—Colombian|
|8.| Mother's qualification <br> Father's qualification | The qualification of the student's mother and father | 1—Secondary Education—12th Year of Schooling or Equivalent <br>2—Higher Education—bachelor’s degree<br>3—Higher Education—degree<br>4—Higher Education—master’s degree<br>5—Higher Education—doctorate<br>6—Frequency of Higher Education<br>7—12th Year of Schooling—not completed<br>8—11th Year of Schooling—not completed<br>9—7th Year (Old)<br>10—Other—11th Year of Schooling<br>11—2nd year complementary high school course<br>12—10th Year of Schooling<br>13—General commerce course<br>14—Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent<br>15—Complementary High School Course<br>16—Technical-professional course<br>17—Complementary High School Course—not concluded<br>18—7th year of schooling<br>19—2nd cycle of the general high school course<br>20—9th Year of Schooling—not completed<br>21—8th year of schooling<br>22—General Course of Administration and Commerce<br>23—Supplementary Accounting and Administration<br>24—Unknown<br>25—Cannot read or write<br>26—Can read without having a 4th year of schooling<br>27—Basic education 1st cycle (4th/5th year) or equivalent<br>28—Basic Education 2nd Cycle (6th/7th/8th Year) or equivalent<br>29—Technological specialization course<br>30—Higher education—degree (1st cycle)<br>31—Specialized higher studies course<br>32—Professional higher technical course<br>33—Higher Education—master’s degree (2nd cycle)<br>34—Higher Education—doctorate (3rd cycle)|
|9.| Mother's occupation <br> Father's occupation | The occupation of the student's Mother and Father |1—Student<br>2—Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers<br>3—Specialists in Intellectual and Scientific Activities<br>4—Intermediate Level Technicians and Professions<br>5—Administrative staff<br>6—Personal Services, Security and Safety Workers, and Sellers<br>7—Farmers and Skilled Workers in Agriculture, Fisheries,and Forestry<br>8—Skilled Workers in Industry, Construction, and Craftsmen<br>9—Installation and Machine Operators and Assembly Workers<br>10—Unskilled Workers<br>11—Armed Forces Professions<br>12—Other Situation; 13—(blank)<br>14—Armed Forces Officers<br>15—Armed Forces Sergeants<br>16—Other Armed Forces personnel<br>17—Directors of administrative and commercial services<br>18—Hotel, catering, trade, and other services directors<br>19—Specialists in the physical sciences, mathematics, engineering,and related techniques<br>20—Health professionals<br>21—Teachers<br>22—Specialists in finance, accounting, administrative organization,and public and commercial relations<br>23—Intermediate level science and engineering techniciansand professions<br>24—Technicians and professionals of intermediate level of health<br>25—Intermediate level technicians from legal, social, sports, cultural,and similar services<br>26—Information and communication technology technicians<br>27—Office workers, secretaries in general, and data processing operators<br>28—Data, accounting, statistical, financial services, and registry-related operators<br>29—Other administrative support staff<br>30—Personal service workers<br>31—Sellers<br>32—Personal care workers and the like<br>33—Protection and security services personnel<br>34—Market-oriented farmers and skilled agricultural and animal production workers<br>35—Farmers, livestock keepers, fishermen, hunters and gatherers,and subsistence<br>36—Skilled construction workers and the like, except electricians<br>37—Skilled workers in metallurgy, metalworking, and similar<br>38—Skilled workers in electricity and electronics<br>39—Workers in food processing, woodworking, and clothing and other industries and crafts<br>40—Fixed plant and machine operators<br>41—Assembly workers<br>42—Vehicle drivers and mobile equipment operators<br>43—Unskilled workers in agriculture, animal production, and fisheries and forestry<br>44—Unskilled workers in extractive industry, construction,manufacturing, and transport<br>45—Meal preparation assistants<br>46—Street vendors (except food) and street service providers|
|10.| Displaced |  Whether the student is a displaced person | 1—yes <br> 0—no|
|11.| Educational special needs | Whether the student has any special educational needs | 1—yes <br> 0—no |
|12.| Debtor | Whether the student is a debtor or not |1—yes <br> 0—no |
|13.| Tuition fees up to date | Whether the student's tuition fees are up to date | 1—yes <br> 0—no|
|14.| Gender | The gender of the student |1—male <br> 0—female |
|15.| Scholarship holder | Whether the student is a scholarship holder |1—yes <br> 0—no|
|16.| Age at enrollment | The age of the student at the time of enrollment | |
|17.| International | Whether the student is an international student |1—yes <br> 0—no |
|18.| Curricular units 1st & 2nd sem (credited) | The number of curricular units credited by the student in the first and second semester | |
|19.| Curricular units 1st & 2nd sem (enrolled) | The number of curricular units enrolled by the student in the first and second semester | |
|20.| Curricular units 1st & 2nd sem (evaluations) | The number of curricular units evaluated by the student in the first and second semester | |
|21.| Curricular units 1st & 2nd sem (approved) | The number of curricular units approved by the student in the first and second semester | |
|22.| Curricular units 1st & 2nd sem (grade) | The number of curricular units grade by the student in the first and second semester | |
|23.| Unemployment rate | The Unemployment rate % | |
|24.| Inflation rate | The Inflation rate % | |
|25.| GDP | GDP per capita (USD) | |
|26.| Target | Status of the student |Graduate <br> Dropout<br> Enrolled |



