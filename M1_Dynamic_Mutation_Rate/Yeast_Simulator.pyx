###################################################################################################################
############################################# MAIN PROGRAM ########################################################
###################################################################################################################
# MAIN CONCEPT: Initial population contains n(Ex.n=10) number of yeast cells and each cell divides at different
# time point (Ex. bw 78-108 mins). When the time goes on, each cell divides at its particular time point.
# Some cells grows faster(78 mins) and some are slower(102 mins). Cell growth will be stopped when the desird
# population size is reached. And then n number of cells will be randomly sampled(bottleneck) and let it grow again.
###################################################################################################################
########################################### IMPORTANT VARIABLES ###################################################
# Genotypes structure goes like this : {id:[next division time, Cell division time, No. of generations,}
# genotypes - contains all the information about the each individual
# mutations - All the mutation structure, beneficial_mutations - contains the IDs of beneficial mutations
# Cell_mutations - cell id and the mutations it carries, gene_duplications - structures of gene duplications.
# cell_gene_duplications - cell id and the gene duplications, gene_deletions - structures of gene deletions
# cell_gene_deletions - cell id and the gene deletions ids, mutations_fitness - mutation id and fitness
# gene_duplication_fitness - gene duplication ID and fitness, gene_deletion_fitness - gene deletion id and fitness.
####################################################################################################################
####################################################################################################################
from __future__ import division
import sys
import cProfile
import scipy.stats
import random
from copy import deepcopy
from numpy import *
import datetime
################################# User defind modules ######################################################
import BackCrossing
#import Artificial_Cells
import Yeast_Simulator_Subprograms
#import Yeast_Simulator_Cell_Division
############################################################################################################
####################### Global Variables - Constants ###############################################################
# Refer the coding documents for all the values use here.
number_of_bottlenecks = 50
# Number of generations before bottleneck.
# Which means the average number of cell divisions
# for the cells in initial population.
number_of_generations = 5
# Min and max cell division time for the cells
# cell_division_min = 78
# cell_division_max = 102
# Use this value for stressor medium.
#cell_division_min = 78
# Mean cell division time of the WT cells in normal medium
cell_division_min = 90
# Maximum cell divison time of the cells in Stressor medium.
cell_division_max = 180
# Maximum Number of times a cell can divide.
cell_max_age = 16
# Number of chromosomes.
n_chr = 16
# Number of Genes in Yeast Genome.
n_genes = 5802
# Mutation rate is 4 out of 1000.
Task_id = sys.argv[1]

decay_constant = 2 #float(sys.argv[2])
mutation_rate =  float(sys.argv[2])
mutation_shape = float(sys.argv[3])
mutation_scale = float(sys.argv[4])
# Beneficial Mutation rate is 88 out of 1000 in Genome wide mutation rate.
beneficial_mutation_rate = float(sys.argv[5]) #60 #0.088
fitness_affecting_mutation_rate = float(sys.argv[6])
beneficial_mutation_fitness_min = 5  # 5%
beneficial_mutation_fitness_max = 10 # 10%
# This is rate is per gene per generation.
# gene_duplication_rate = 0.0000034
# gene_duplication_rate = 0.0197268 #float(sys.argv[4])
gene_duplication_rate = 0 #0.0195942 #float(sys.argv[4])
gene_duplication_shape = 0.4 #2 #float(sys.argv[5])
gene_duplication_scale = 0.2 #1 #float(sys.argv[6])
gene_duplication_fitness = 0.5
gene_duplication_beneficial = 100 #float(sys.argv[7])
gene_duplication_neutral = 0 #float(sys.argv[8])
gene_duplication_deleterious = 0 #float(sys.argv[9])
gene_duplication_fitness_truncation = 16 # float(sys.argv[10])
# Non-Random Sasmpling
# Faster cells proportion 
fcp = 40
# Medium cells proportion
mcp = 20
# Slower cells proportion
scp = 40
# Gene Duplication Rate is 1218 out of 100000
# Value is converted from 1 gene rate to 5802 genes.
gene_deletion_rate = 0.0121842 # Out of 10000000
# Number of gene duplications
n_gene_duplications = 0
# Number of gene deletions.
n_gene_deletions = 0
# Beneficial Mutation rate is 575 out of 10000.
# beneficial_mutation_rate = 0.0575
# Number of individuals.
n_individuals = 1000
# cell % of wont viable
cell_wont_survive = 10
# Variables which are used for Testing purpose.
number_of_mitosis_events = 0
total_number_of_mitosis_events = 0
total_population_size = 0

# Number of cells for mitosis is between 20-25
# Number of evolved cells to backcross.
import random
min_sample_size = 20
max_sample_size = 25
n_evolved_cells = random.randrange(min_sample_size,max_sample_size)
# Number of wildtype to backcross.
n_wildtype = n_evolved_cells
n_backcrossing = 10
# Mating Parameters
p_m = 0.90

# To store the wildtype cells.
wildtype_cells = {}
# To store the evolved cells.
evolved_cells = {}

# n_mut will tell the number of mutation events happened.
# tp: total population. tp will tell the population size.

# Age and Cell division time distribution.
# Ex. cell_division_time{age:cellDivisionTime}
"""
cell_division_time = {0:102,1:102,2:98,3:94,
                      4:88,5:84,6:78,7:80,8:82,
                      9:84,10:86,11:88,12:90,
                      13:92,14:94,15:96,16:102}
cell_division_time_distribution = [78,80,82,84,86,
                                   88,90,92,9,94,96,102]
"""
cell_division_time_increase_proportions = [0.0219414690133,0.0223935772857,0.0228448947257,0.0232950247887,0.0237435751105,0.0241901588628,0.0246343960682,0.0250759148643,
                                           0.0255143527094,0.0259493575199,0.0263805887355,0.0268077183016,0.0272304315679,0.0276484280956,0.0280614223718,0.0284691444277]

##############################################################################################################
####                         Global variables                                                        #########
##############################################################################################################
# Genotypes is a dict which contains all the details of each cell.
genotypes = {} 
# Contains all the mutations
mutations = []
# Contains beneficial Mutations
beneficial_mutations = []
# Contains beneficial mutation's fitness.
# Index will be the same for beneficial_mutations variable.
mutations_fitness = []
# Cell ID and the beneficial Mutation IDs
cell_beneficial_mutations = []
# Contains the haplotype of gene duplications
gene_duplications = []
# Contains the gene duplication fitness.
# Index will be the same as gene_duplications variable.
gene_duplications_fitness = []
# Contains the cell ID and gene duplication ids.
cell_gene_duplications = []
# Contains Gene deletion haplotypes
gene_deletions = []
# Contains gene deletion fitness
# Index is same as gene_deletions variable.
gene_deletions_fitness = []
# Contains cell id and the gene deletion IDs
cell_gene_deletions = []
n_individuals_with_mutation = 0
# This variables will be used in Mutation freq calculalation.
bottleneck_positions = {}
bottleneck_positions[0] = 0
# Cell group based on their cell division time.
cell_groups = {}
# Total number of mutation events.
n_mut = 0

cell_mutations = array([])

# desired population size.
dps = n_individuals * 2**number_of_generations


#############################################################################################################
# Subprograms
#############################################################################################################
def make_fitness_proportions(fitness,MP):
    import random as r
    n1 = int(len(fitness) * (100-MP)/100)
    n2 = len(fitness)
    pop = xrange(n2)
    sampled = r.sample(pop,n1)
    for i in sampled:
        fitness[i] *= -1

    return fitness
    
def scale_fitness_effects(fitness):
    fitness_scale = make_scales()
    for i in range(len(fitness)):
        id = "%.4f" % round(fitness[i],4)
        fitness[i] = fitness_scale[id]
    
    return fitness
    
def make_scales():
    scale = {}
    alpha = 0
    for i in range(50000):
        P = -math.log(2) * alpha 
        CDT1 = 90 * math.exp(P) + 90
        CDT2 = 180 - 90*alpha
        new_alpha = alpha
        if CDT1 < CDT2:
            new_alpha = find_new_alpha(CDT1,CDT2,alpha)

        alpha1 = "%.4f" % round(alpha,4)
        
        scale[alpha1] = new_alpha
        alpha += 0.0001
        
    return scale


def covert_to_minutes(fitness):
    fitness_mins = []
    for i in fitness:
        fitness_mins.append(calculate_cell_division_time_1(i))
    
    return fitness_mins

def covert_to_minutes_1(fitness):
    fitness_mins = []
    for i in fitness:
        fitness_mins.append(calculate_cell_division_time_2(i))
    
    return fitness_mins

def calculate_cell_division_time_1(alpha):
    cell_division_max = 180
    cell_division_min = 90
    change_CDT = (cell_division_max-cell_division_min) * alpha
    
    if change_CDT >= 90:
        change_CDT = 89
        
    return change_CDT


def calculate_cell_division_time_2(alpha):
    cell_division_max = 660
    cell_division_min = 300
    change_CDT = (cell_division_max-cell_division_min) * alpha
    
    if change_CDT >= 359:
        change_CDT = 359
        
    return change_CDT
    
    
def truncate_fitness_effects(fitness,trunc):
    new_fitness = []
    for i in fitness:
        if i < trunc:
            new_fitness.append(i)
    
    return new_fitness


def calculate_cell_division_time(current_CDT,alpha):
    #print "current_CDT:",current_CDT
    #print "alpha:",alpha
    
    # Cell Division time to "Alpha"-refer the coding documents.
    cell_division_max = 180
    cell_division_min = 126
    CDT_change = 0
    if alpha > 0:
        CDT_change = (current_CDT - cell_division_min)/(cell_division_max - cell_division_min)
        CDT_change *= alpha
        new_CDT = current_CDT - CDT_change
    else:
        new_CDT = current_CDT + abs(alpha)
    
    #print "new_CDT:",new_CDT
    
    return  new_CDT

def calculate_Lag_time(current_CDT,alpha):
    # Cell Division time to "Alpha"-refer the coding documents.
    cell_division_max = 660
    cell_division_min = 300
    CDT_change = 0
    if alpha > 0:
        CDT_change = (current_CDT - cell_division_min)/(cell_division_max - cell_division_min)
        CDT_change *= alpha
        new_CDT = current_CDT - CDT_change
    else:
        new_CDT = current_CDT + abs(alpha)

    return  new_CDT
    
    
#############################################################################################################
#   Initializing the Fitnesss effects and haplotype of genetic variants
#############################################################################################################
mutation_fitness_random_fitness = scipy.stats.gamma.rvs(mutation_shape,loc=0,scale=float(1)/float(mutation_scale),size=dps*0.25* number_of_bottlenecks)
mutation_fitness_random_fitness = scale_fitness_effects(mutation_fitness_random_fitness)

mutation_fitness_random_fitness_Rate = covert_to_minutes(mutation_fitness_random_fitness)
mutation_fitness_random_fitness_Lag = covert_to_minutes_1(mutation_fitness_random_fitness)
  
mutation_fitness_random_fitness_Rate = truncate_fitness_effects(mutation_fitness_random_fitness_Rate,54)
mutation_fitness_random_fitness_Lag = truncate_fitness_effects(mutation_fitness_random_fitness_Lag,359)

mutation_fitness_Rate  = make_fitness_proportions(mutation_fitness_random_fitness_Rate,beneficial_mutation_rate)
mutation_fitness_Lag  = make_fitness_proportions(mutation_fitness_random_fitness_Lag,beneficial_mutation_rate)

#############################################################################################################
# Main Program: Contains selection experiment simulation as well as Automated Backcrossings
#############################################################################################################
def Yeast_lab():
    import Yeast_Simulator_Subprograms
    import BackCrossing
    N_HAP1 = 0;N_HAP2 = 0;N_HAP3 = 0

    print "CDT\t",0,"\t",805,"\t","Lag"
    print "CDT\t",0,"\t",162,"\t","Rate"

    # WT LAG TIME
    LT = 805
    
    # WT RATE TIME
    RT = 162
    
    # Mut LAG TIME
    M_LT = 300
    
    # Mut RATE TIME
    M_RT = 126
    
    # Initialize the cell Population
    genotypes,PM,NN,CRT,CLT = Yeast_Simulator_Subprograms.initialize_population(n_individuals,number_of_generations,LT,RT,M_LT)    
    
    # Grouping the cells based on their cell division time.
    cell_groups =  Yeast_Simulator_Subprograms.group_cells(genotypes,n_individuals)

    # Local variables for the subroutine.     
    #mutations_fitness = {}; PM_strand = {};
    
    # Opening files for storing the HAPLOTYPE OF POINT MUTATIONS AND GENE DUPLICATIONS.
    point_mutation = open("Point_Mutations.txt","w")
    Fname = "PM_Track_" + str(Task_id) + ".csv"
    PM_Track = open(Fname,"w")
    
    import Yeast_Simulator_Cell_Division
    n_mut = 0;
    # Cell_Haplotype = {}
    CH =    {}
    N_CH = {}
    MID_Hap = {}
    for i in xrange(n_individuals):
        CH[i] = "W"
        N_CH[i] = 0
    
    # Ploidy = 1 : Haploid, ploidy = 2: Diploid
    ploidy = 1; div_type = 1; s_size = n_individuals; fix_cal = 1
    # Calling the function to allow the cells to divide. Mutations, gene deletions and gene duplications will be introduced during cell division.
    (genotypes,PM,PM_fitness,PM_strand,n_mut) = Yeast_Simulator_Cell_Division.asymmetrical_cell_division(genotypes,cell_groups,
                    mutation_fitness_Rate,mutation_fitness_Lag,PM,number_of_generations,number_of_bottlenecks,n_individuals,n_mut,point_mutation,ploidy,"ASE",div_type,s_size,fix_cal,PM_Track,NN,M_RT,LT,M_LT,CRT,CLT,N_HAP1,N_HAP2,N_HAP3,CH,N_CH,MID_Hap)
            
    print "END_OF_ASE"
    PM_Track.close()
    # Closing the HAPLOTYPE files
    
    sys.exit()

#############################################################################################################
#############################################################################################################

#cProfile.run("Yeast_lab()")


# This is to run this module normally like "python Yeast_Simulator.py"
if __name__ == "__main__":
    import sys
    Yeast_lab()
