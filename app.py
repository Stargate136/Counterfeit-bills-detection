import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def train_model():
    df = pd.read_csv("https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/billets.csv", sep=";")
    df.dropna(inplace=True)

    X = df[["length", "margin_low", "margin_up"]]
    y = df["is_genuine"]

    # Pas besoin de split, les performances ont été évaluées lors des tests
    log_reg = LogisticRegression().fit(X, y)
    
    return log_reg

def predict(model, *features):
    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    pred_proba = proba[model.classes_ == pred][0]

    return pred, pred_proba * 100

def main():
    model = train_model()

    st.title("DETECTION DES FAUX BILLETS")

    with st.form("features"):
    
        # Champs du formulaire
        length = st.number_input("Longueur du billet (en mm)")
        margin_low = st.number_input("Marge en bas du billet (en mm)")
        margin_up = st.number_input("Marge en haut du billet (en mm)")
        
        # Bouton de soumission
        submit_button = st.form_submit_button(label='Vérification')

        # Vérifications après avoir cliqué sur le bouton de soumission
        if submit_button:
            if not length:
                st.warning("Veuillez entrer la longueur du billet")
            if not margin_low:
                st.warning("Veuillez entrer la marge en bas du billet")
            if not margin_up:
                st.warning("Veuillez entrer la marge en haut du billet")
            else:
                pred, proba = predict(model, length, margin_low, margin_up)
                if pred:
                    st.success(f"Il s'agit d'un vrai billet à {proba:.2f} %")
                else:
                    st.warning(f"Il s'agit d'un faux billet à {proba:.2f} %")
         

if __name__ == "__main__":
	main()