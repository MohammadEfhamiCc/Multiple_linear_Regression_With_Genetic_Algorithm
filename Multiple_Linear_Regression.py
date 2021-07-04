__author__ = "Mohammad Efhami Sisi"

from operator import le
from typing import final
from joblib.logger import PrintTime
import pandas as pd
import seaborn as sns
import numpy as np
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from operator import attrgetter
import sys
from collections.abc import Sequence
from itertools import repeat
from random import gauss, shuffle
import matplotlib.pyplot as plt
import os
import getopt
import time
import seaborn as sea

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.realpath(THIS_DIR))

regressor = LinearRegression()
scaller = MinMaxScaler()

class Multiple_Linear_Regression:
    # Please keep it in mind, in your usecase, change carefully the line 37 for the name of input file,
    # 40, for name of attributes or features, write them carefully,41, for the name of class, the targer feature or attribute,
    # we want to predict the value of 'DEATH_EVENT' feature.
    def __init__(self,file_exist,generation_arg,selection_parameter):
        if file_exist==True:   
            self.dataset_address = "/".join([THIS_DIR, "data.csv"])
            self.dataset=pd.read_csv(self.dataset_address)
            self.X = self.dataset[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
            self.y = self.dataset['DEATH_EVENT']
            self.data = scaller.fit_transform(self.X)
            self.X_test=[]
            self.y_test=[]
            self.fitness_of_minimum_mse_in_current_generation=10
            self.generation_termination=generation_arg
            self.tournament_selection_parameter=selection_parameter
            self.initial_population=self.generate_initial_population()
        else:
            message='\nthe dataset does not exist, please chech the address of your dataset,\
              if you are sure, please check the name of dataset, or even check the suffix,\
       for example, .csv is diffrent from .xlsx, so be careful'
            sys.exit(message)
    
    # my initial population consist of 100 chromosomes, half of them have been generated with training a regression model and take the 
    # coefficients and internept of it, and the other half are made randomly.
    # if you had plane to change the number of chromosomes in each generation, you should do it here, or even the percentage of randomly generated 
    # chromosomes and the values the have been obtained from our regression model, all of them take place in this function
    def generate_initial_population(self):
        coef=[]
        random_coef=list()
        random_coef_temp=list()
        # generating first half of the initial population
        for i in range(50):
            X_train, self.X_test, y_train, self.y_test = train_test_split(self.data, self.y, test_size=0.2, train_size=0.8, random_state=i)
            regressor.fit(X_train, y_train)
            coef.append(regressor.coef_.tolist())
            coef[i].insert(0,regressor.intercept_)
            print("the {}th chromosome has been generated for initial population.".format((i+1)))

        # for i in range(650):
        #     a = random.choice(range(10, 1000))
        #     x=random.uniform(-1,1)/a
        #     random_coef_temp.append(x)
        #     if((i+1)%13==0):
        #         random_coef.append(random_coef_temp)
        #         random_coef_temp=[]
        #         print("the {}th chromosome has been generated for initial population.".format(d))
        #         d+=1

        d=51
        # generating the other half of initial population
        for i in range(50):
            for z in range(13):
                a = random.choice(range(10, 1000))
                x=random.uniform(-1,1)/a
                random_coef_temp.append(x)
            random_coef.append(random_coef_temp)
            random_coef_temp=[]
            print("the {}th chromosome has been generated for initial population.".format(d))
            d+=1


        final_population=[]
        final_population = coef + random_coef
        shuffled_population=final_population

        # shuffeling the initial population to get a well mixed initial population
        for c in range(20):
            shuffled_population=random.sample(shuffled_population,len(shuffled_population))
        print("\n")
        print("the size of initial population is: {}".format(len(shuffled_population)))
        print("\n")
        return self.genetic_algorithm(shuffled_population,self.generation_termination)

    # calculating fitness of chromosome, how well is our individual/chromosome, in this situation, lower is better
    @staticmethod
    def fiteness(chromosome, x_test, y_test):
        regressor.intercept_= chromosome[0]
        regressor.coef_= np.array(chromosome[1:13])
        y_pred = regressor.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        return mse
    
    # one of the well known algorithms for selecting chromosome/chromosome is tounament selection, with this 
    # algorithm even chromosome with Undesirable fitness has chance to reproduction.
    def tournament_selection(self,chromosomes,k):
        random_loc=[]
        random_chro_fitness=[]
        for c in range(k):
            random_loc.append(chromosomes[random.randint(0, len(chromosomes)-1)])
        for b in range(k):
            random_chro_fitness.append(self.fiteness(random_loc[b], self.X_test, self.y_test))
        min_fitness_loc=random_chro_fitness.index(min(random_chro_fitness))
        return random_loc[min_fitness_loc]

    # uniform cross over for combining two choromosome with each other, for every gene the algorithm decides whether it
    # would exchanged or not
    @staticmethod
    def uniform_cross_over(chromosome1, chromosome2, px):
        size = min(len(chromosome1), len(chromosome2))
        for i in range(size):
            if random.random() < px:
                chromosome1[i], chromosome2[i] = chromosome2[i], chromosome1[i]
        return chromosome1, chromosome2

    # gaussian mutation function to mutate a given chromosome, for every gene in chromosome it would generate a random random from gaussian distribution
    # and sum it with the value of the gene, for every gene in chromosome this process would be done
    @staticmethod
    def gaussian_mutation(chromosome, mean, sigma):
        chr = []
        for e in range(13):
            new_value=0
            gaussian_value = random.gauss(mean,sigma)
            new_value=chromosome[e]+gaussian_value
            chr.append(new_value)
        return chr

    # main dunciton that call the other functions to get the work done
    def genetic_algorithm(self,chromosomes,generation_termination):
        current_generation_chromosomes=chromosomes
        count_of_current_chromosomes=0
        parents_for_CO_M=[]
        next_generation_chromosomes=[]
        # stores the cross overd chromosomes
        CO_next_generation_chromosomes=[] 
        # stores the mutated chromosomes
        MU_next_generation_chromosomes=[] 
        fitness_of_each_chromosomes_in_current_generation=[]
        generation=0
        final_chromosomes=[]
        all_generations=[]

        print("the value of ' MSE ' for each generation are listed below:")
        print("\n")
        for r in range(generation_termination):# for every Generation and also our stopping condition
            for s in range(50):                # for populations, in each loop a total number of two child would be made, 50*2=100=lengthOfInitialPopulation
                for t in range(2):             # for selections
                    #in each loop a chromosome would be selected to with the other selected chromosome they would conduct cross over and mutation
                    # and at the end of this process they would transferred to next generation 
                    parents_for_CO_M.append(self.tournament_selection(current_generation_chromosomes,self.tournament_selection_parameter))
                
                if generation==0:
                    histogeram_list=[]
                    histogeram_list.append(current_generation_chromosomes)

                temp_tuple=self.uniform_cross_over(parents_for_CO_M[0],parents_for_CO_M[1],0.5)
                parents_for_CO_M=[]
                list1=[]
                for x in range(len(temp_tuple)):
                    list1.append(temp_tuple[x])
                CO_next_generation_chromosomes.append(list1[0])
                CO_next_generation_chromosomes.append(list1[1])
                list1=[]
                MU_next_generation_chromosomes.append(self.gaussian_mutation(CO_next_generation_chromosomes[0], 0, 0.01))
                MU_next_generation_chromosomes.append(self.gaussian_mutation(CO_next_generation_chromosomes[1], 0, 0.01))
                CO_next_generation_chromosomes=[]
                next_generation_chromosomes.append(MU_next_generation_chromosomes[0])
                next_generation_chromosomes.append(MU_next_generation_chromosomes[1])
                MU_next_generation_chromosomes=[]
                count_of_current_chromosomes=count_of_current_chromosomes+1
            for t in range(len(current_generation_chromosomes)):
                fitness_of_each_chromosomes_in_current_generation.append(self.fiteness(current_generation_chromosomes[t], self.X_test,self.y_test))
            sign=''
            if self.fitness_of_minimum_mse_in_current_generation>min(fitness_of_each_chromosomes_in_current_generation):
                sign='-'
            if self.fitness_of_minimum_mse_in_current_generation<min(fitness_of_each_chromosomes_in_current_generation):
                sign='+'
            if self.fitness_of_minimum_mse_in_current_generation==min(fitness_of_each_chromosomes_in_current_generation):
                sign='='
            #sign=''
            self.fitness_of_minimum_mse_in_current_generation=min(fitness_of_each_chromosomes_in_current_generation)
            print("in the {}th generation, the value of minimum ' MSE ' is :{} {}".format(generation,self.fitness_of_minimum_mse_in_current_generation,sign))
            final_chromosomes=current_generation_chromosomes
            all_generations.append(current_generation_chromosomes)
            current_generation_chromosomes=[]
            current_generation_chromosomes=next_generation_chromosomes
            next_generation_chromosomes=[]
            fitness_of_each_chromosomes_in_current_generation=[]
            generation+=1

        temp_fitnesses=[]
        current_generation_chromosomes=[]
        for i in range(len(final_chromosomes)):
            temp_fitnesses.append(self.fiteness(final_chromosomes[i],self.X_test,self.y_test))
  
        minpos = temp_fitnesses.index(min(temp_fitnesses))
        maxpos = temp_fitnesses.index(max(temp_fitnesses)) 

        print("\n")
        print("the max and min of ' MSE ' at the last generation are:\n")
        print ("The maximum is at position.", maxpos)  
        print("the value of maximum MSE is:", max(temp_fitnesses))
        print ("The minimum is at position.", minpos)
        print("the value of minimum MSE is:", min(temp_fitnesses))
        print("\n")
        print("\n")
        print("the intercept and coefficients of maximum chromosome is:", final_chromosomes[maxpos])
        print("\n")
        print("the intercept and coefficients of minimum chromosome is:", final_chromosomes[minpos])
        return all_generations


try:
    opts, args = getopt.getopt(sys.argv[1:], 'g:k:')
except (getopt.GetoptError, getopt.err):
    sys.exit(2)
for o, a in opts:
    if o == '-g':# the termination condition to stop the program, one of the three rules predefined condition to stop the program,
                 # first is when we reach a certain value in fitness,second is when we do not have improvement in fitness in several consevutive generation,
                 # and the last one is when we reach to a certain number of generation, in this program we used the last condition.
        generation_arg = int(a)
    if o == '-k':# parameter for tournament selection function
        selection_parameter = int(a)

dataset_address = "/".join([THIS_DIR, "data.csv"])

file_exist=os.path.exists(dataset_address)

MLR_initiate=Multiple_Linear_Regression(file_exist,generation_arg,selection_parameter)