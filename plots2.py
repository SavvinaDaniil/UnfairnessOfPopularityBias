import matplotlib.pyplot as plt
import numpy as np
def plot_GAP_algorithm_results(low_gap_vals, medium_gap_vals, high_gap_vals, item_col, save = False, addition = ""):
    barWidth = 0.1

    # set height of bar
    bars1 = [low_gap_vals[0], medium_gap_vals[0], high_gap_vals[0]]
    bars2 = [low_gap_vals[1], medium_gap_vals[1], high_gap_vals[1]]
    bars3 = [low_gap_vals[2], medium_gap_vals[2], high_gap_vals[2]]
    bars4 = [low_gap_vals[3], medium_gap_vals[3], high_gap_vals[3]]
    bars5 = [low_gap_vals[4], medium_gap_vals[4], high_gap_vals[4]]
    bars6 = [low_gap_vals[5], medium_gap_vals[5], high_gap_vals[5]]
    bars7 = [low_gap_vals[6], medium_gap_vals[6], high_gap_vals[6]]
    #bars8 = [low_gap_vals[7], medium_gap_vals[7], high_gap_vals[7]]
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars3))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]
    #r8 = [x + barWidth for x in r7]
    ax = plt.axes()
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['left'].set_zorder(0)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    
    ax.set_facecolor("aliceblue")
    # Make the plot
    plt.bar(r1, bars1, width=barWidth, label='Random')
    plt.bar(r2, bars2, width=barWidth, label='MostPopular')
    plt.bar(r3, bars3, width=barWidth, label='UserItemAvg')
    plt.bar(r4, bars4, width=barWidth, label='UserKNN')
    #plt.bar(r5, bars5, width=barWidth, label='ItemKNN')
    plt.bar(r5, bars5, width=barWidth, label='UserKNNAvg')
    plt.bar(r6, bars6, width=barWidth, label='NMF')
    plt.bar(r7, bars7, width=barWidth, label='SVD')
    
    # Add xticks on the middle of the group bars + show legend
    plt.xlabel('User group', fontsize='15')
    plt.ylabel('% $\Delta$ GAP', fontsize='15')
    plt.xticks([r + barWidth for r in range(len(bars3))], ['LowMS', 'MedMS', 'HighMS'], fontsize='13')
    plt.yticks(fontsize='13')
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., framealpha=1, fontsize='15')
    #plt.savefig('data/ECIR/gap_analysis.png', dpi=300, bbox_inches='tight')
    if save:
        plt.savefig('graphs/'+item_col+"_group_results"+addition+".png",  bbox_inches='tight')
    plt.show(block=True)