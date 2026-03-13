import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns

def extract_weights_from_subdirs(parent_dir):
    all_assets_data = {}

    # On parcourt chaque sous-dossier dans le répertoire parent
    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)

        # On ne traite que les dossiers qui concernent l'allocation
        if os.path.isdir(folder_path) and "Allocation_Portfolio_Weights" in folder:
            asset_name = folder.split('_')[-1]  # Récupère le nom (ex: AMZN)

            # Charger l'accumulateur pour ce dossier spécifique
            event_acc = EventAccumulator(folder_path)
            event_acc.Reload()

            # Dans ces sous-dossiers, le tag est souvent simplifié
            # On cherche le tag de scalaire disponible
            tags = event_acc.Tags()['scalars']
            if tags:
                tag = tags[0]  # Généralement il n'y en a qu'un par dossier
                scalars = event_acc.Scalars(tag)
                steps = [e.step for e in scalars]
                values = [e.value for e in scalars]

                all_assets_data[asset_name] = pd.Series(values, index=steps)

    df = pd.DataFrame(all_assets_data)
    df.index.name = 'step'
    return df

def plot_weights(df_weights):
    plt.figure(figsize=(15, 8))
    # On transpose pour avoir les actifs en Y et le temps en X
    sns.heatmap(df_weights.T, cmap="rocket", cbar_kws={'label': 'Poids du portefeuille'})
    plt.title("Intensité de l'allocation Borey-Alpha par actif")
    plt.xlabel("Steps")
    plt.ylabel("Actifs")
    plt.show()



if __name__ == "__main__":
    log_path = "../tensorboard_logs/test_results_PMPT_10M/"
    df_weights = extract_weights_from_subdirs(log_path)
    top_assets = df_weights.mean().sort_values(ascending=True).index
    df_weights_sorted = df_weights[top_assets]
    plot_weights(df_weights_sorted)
    #TODO