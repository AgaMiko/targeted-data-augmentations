import pandas as pd

# group by model and aug_type
cbi_results = pd.read_csv('cbi_results.tsv')
general_aggregated = cbi_results.groupby(by=["model","aug_type", "p" ]).mean()

# group by model and aug_type
aug_type = cbi_results.groupby(by=["aug_type" ]).mean()

# group by p and aug type
p_aug_type = cbi_results.groupby(by=["aug_type", "p" ]).mean()

# which mask is the worst
worst_mask = cbi_results.groupby(by=["aug_type",  "mask_nr"]).mean()

# standard deviation
std = cbi_results.groupby(by=["model","aug_type", "p"]).std()

# descriptive stats 
stats = cbi_results.groupby(by=["model","aug_type", "p"]).describe()


with pd.ExcelWriter('aggregated_cbi_results.xlsx') as writer:  
    general_aggregated.to_excel(writer, sheet_name='path and aug_type')
    aug_type.to_excel(writer, sheet_name='aug_type')
    p_aug_type.to_excel(writer, sheet_name='p_aug_type')
    worst_mask.to_excel(writer, sheet_name='worst_mask')
    std.to_excel(writer, sheet_name='std')
    stats.to_excel(writer, sheet_name='descriptive stats')