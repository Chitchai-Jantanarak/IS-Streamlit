import os
import pandas as pd

def data_cleansing() -> pd:

    def processing(df) -> pd:
        df = df.drop( columns = ["Comments"] ) # Drop Unnecessary Attr.
        df = df.dropna(subset = ["Sex"])       # Drop Na

        # Fill Na based on Species <Mapping>
        species_mean15 = df.groupby('Species')['Delta 15 N (o/oo)'].mean()
        species_mean_dict15 = species_mean15.to_dict()

        species_mean13 = df.groupby('Species')['Delta 13 C (o/oo)'].mean()
        species_mean_dict13 = species_mean13.to_dict()

        # Loop dict & set
        for s, val in species_mean_dict15.items():
            df.loc[(df['Species'] == s) & (df['Delta 15 N (o/oo)'].isna()), 'Delta 15 N (o/oo)'] = val

        for s, val in species_mean_dict13.items():
            df.loc[(df['Species'] == s) & (df['Delta 13 C (o/oo)'].isna()), 'Delta 13 C (o/oo)'] = val

        # Found unknown value on 'Sex'
        df = df[df['Sex'] != '.']

        # ENCODING
        df.loc[:, 'Clutch Completion'] = df['Clutch Completion'].map({'Yes': 1, 'No': 0})
        df.loc[:, 'Sex'] = df['Sex'].map({'MALE': 1, 'FEMALE': 0})

        return df
    
    # Configurate the file path from here kub :)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, 'penguins_lter.csv')
    df = pd.read_csv(file_path)

    return processing(df)