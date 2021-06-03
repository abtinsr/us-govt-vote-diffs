################################################
### IMPORT PACKAGES
################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

################################################
### CALCULATE VOTE SHARES PER POLITICAL RACE
################################################
def vote_shares(election_data, label): 
    
    total_votes_by_state_year = (
        election_data.copy()
        .groupby(['state', 'year'])
        .aggregate({'candidatevotes':'sum'})
        .reset_index()
        .rename(columns={'candidatevotes': f'{label.upper()}_total_votes'})
    )

    party_votes_by_state_year = (
        election_data.copy()
        .groupby(['state', 'year', 'party_simplified'])
        .aggregate({'candidatevotes':'sum'})
        .reset_index()
        .rename(columns={'candidatevotes': f'{label.upper()}_candidate_votes'})
    )

    df_merged = pd.merge(party_votes_by_state_year, total_votes_by_state_year)

    df_merged[f'{label.upper()}_share_of_votes'] = round(df_merged[f'{label.upper()}_candidate_votes'] / df_merged[f'{label.upper()}_total_votes'],3)
    
    return(df_merged)

################################################
### HANDLE CONGRESSIONAL SPECIAL ELECTIONS
################################################
# Both the senate and the house races could be so called "special elections," 
# which need to be treated separately so as to not mess up the vote shares 
# for a particular state and year. We want to avoid aggregating votes that 
# are not part of the same election.

def join_regular_special_elections(election_data, label): 
    
    df = pd.DataFrame()
    
    for i in [True, False]:
        selected_races = election_data[election_data['special']==i]
        selected_races = vote_shares(election_data=selected_races, label=label)
        df = df.append(selected_races)
    
    return(df)

################################################
### JOIN DATA FROM PRESIDENTIAL AND CONGRESSIONAL RACES
################################################
def votes_by_state_party_year(house_data, senate_data, pres_data):
    
    # Both the senate and the house races could be so called "special elections," which need to be treated separately
    # so as to not mess up the vote shares for a particular state and year. 
    
    ########################
    # HOUSE RACE VOTES
    ########################
    house = join_regular_special_elections(election_data=house_data, label='house')
    
    ########################
    # SENATE RACE VOTES
    ########################
    senate = join_regular_special_elections(election_data=senate_data, label='senate')
    
    ########################
    # PRESIDENTIAL RACE VOTES
    ########################
    pres = vote_shares(election_data=pres_data, label='pres')
    
    ########################
    # JOIN DATASETS
    ########################
    merged_data = pd.merge(pres, senate, how='left', on=['year', 'state', 'party_simplified'])
    merged_data = pd.merge(merged_data, house, how='left', on=['year', 'state', 'party_simplified'])
    
    merged_data = (
        merged_data
        .assign(PRES_minus_SENATE_diff=lambda x: x['PRES_share_of_votes'] - x['SENATE_share_of_votes'])
        .assign(PRES_minus_HOUSE_diff=lambda x: x['PRES_share_of_votes'] - x['HOUSE_share_of_votes'])
        .assign(SENATE_minus_HOUSE_diff=lambda x: x['SENATE_share_of_votes'] - x['HOUSE_share_of_votes'])
    )

    return(merged_data)


################################################
### VISUALISE DISCREPANCIES ON FEDERAL LEVEL
################################################
def visualise_federal_level(election_data, metric, parties):
    
    viz = (
        election_data
        .query(f'party_simplified in {parties}')
    )
    
    # Edit chamber names for titles and labels. 
    text = metric
    sep = '_minus_'
    chamber1 = text.split(sep, 1)[0]
    chamber2 = text.split(sep, 1)[1].replace("_diff", "")

    # Visualise!
    plt.figure(figsize=(20,10))
    sns.set_context('poster')
    
    ax = sns.lineplot(data=viz, 
                      x='year',
                      y=metric,
                      hue='party_simplified', 
                      palette=['Navy', 'Red'])
    
    # Add labels.
    plt.ylabel(f'{chamber1.upper()} race over/underperformance (versus {chamber2.upper()})')
    plt.xlabel('Year')
    plt.title(f'How much better/worse do the parties perform in {chamber1.upper()} than {chamber2.upper()} races?')
    
    # Add a zero horisontal line - which indicates no difference between races.
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Add tick marks.
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(viz.year.unique()) # https://stackoverflow.com/questions/21910986/why-set-xticks-doesnt-set-the-labels-of-ticks/21911813#21911813
    
    
################################################
### VISUALISE DISCREPANCIES ON STATE LEVEL
################################################
def visualise_state_level(election_data, metric, parties):
    
    viz = (
        election_data
        .query(f'party_simplified in {parties}')
    )
    
    # Edit chamber names for titles and labels. 
    text = metric
    sep = '_minus_'
    chamber1 = text.split(sep, 1)[0]
    chamber2 = text.split(sep, 1)[1].replace("_diff", "")

    plt.figure(figsize=(10,10))
    sns.set_context('paper')

    g = sns.relplot(data=viz, 
                    x='year',
                    y=metric,
                    hue='party_simplified',
                    col='state',
                    kind='line',
                    col_wrap=3, # 2
                    palette=['Navy', 'Red'])

    (
        g
        .map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
        .set_axis_labels("Year", f"{chamber1.upper()} minus {chamber2.upper()} votes (%)")
    )

    for ax in g.axes.flat: # Do this to add the axes to all graphs.
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    ax.set_xticks(viz.year.unique())

    # https://stackoverflow.com/questions/54209895/seaborn-relplot-how-to-control-the-location-of-the-legend-and-add-title
    leg = g._legend
    leg.set_bbox_to_anchor([1, 1]);  # coordinates of lower left of bounding box
    #leg._loc = 2  # if required you can set the loc

    plt.show();
    
    
################################################
### VISUALISE TOTALS PER PARTY
################################################  
def visualise_votes_per_party(election_data, party):
    
    viz = (
        election_data[['year', 'state', 'party_simplified', 'PRES_share_of_votes', 'SENATE_share_of_votes', 'HOUSE_share_of_votes']]
        .query(f'party_simplified in {party}')
        .rename(columns={'PRES_share_of_votes': 'presidential', 
                         'SENATE_share_of_votes': 'senate',
                         'HOUSE_share_of_votes': 'house'})
        .melt(id_vars=['year', 'state', 'party_simplified'], 
              value_vars=['presidential', 'senate', 'house'],
              var_name='race', value_name='share_of_votes')
    )
    
    
    plt.figure(figsize=(20,10))
    sns.set_context('poster')

    ax = sns.lineplot(data=viz, 
                      x='year',
                      y='share_of_votes',
                      hue='race',
                      palette=['Purple', 'Orange', 'Green'])

    plt.ylabel(f'Votes to {party[0]} party (%)')
    plt.xlabel('Year')
    plt.title(f'Votes to {party[0]} party by political race and year')

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_xticks(viz.year.unique()) # https://stackoverflow.com/questions/21910986/why-set-xticks-doesnt-set-the-labels-of-ticks/21911813#21911813