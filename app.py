import numpy as np
import pandas as pd
import streamlit as st
import pickle
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from st_aggrid import AgGrid
from st_aggrid.shared import GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

def main():
    st.title("Automated Data Labelling")
    st.subheader("Malicious URL Detection") 
    st.markdown(
        """
        
        > Let's follow a _six-step_ process:
        - Upload csv file
        - Predict the output
        - Incorrect prediction??? :face_with_rolling_eyes: Correct it yourself :white_check_mark:
        - Save changes
        - Retrain
        - Doubt the predictions? :confused: Hit the explain button


        """
    )
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Predictor API</h2>
    </div>
    """
    file = st.file_uploader('Upload file', type="csv")
    show_file = st.empty()

    if not file:
        show_file.info("Please upload a valid csv file")
        return

    df = pd.read_csv(file)
    # st.dataframe(df)
    # st.info(len(df))

    def predict(df):
        data = df
        data = data.drop("url",1)
        Y = data['label']
        X = data.drop('label', axis=1)
        pickle_in = open("LR_model.pkl","rb")
        clf = pickle.load(pickle_in)
        prediction = clf.predict(X)
        prob = clf.predict_proba(X)
        confidence_score = []
        for i in range(len(prob)):
            if prediction[i]==0:
                confidence_score.append(prob[i][0])
            else:
                confidence_score.append(prob[i][1])
        
        df = df.drop('label', axis=1)
        df['prediction'] = prediction
        df['confidence_score'] = confidence_score
        df['ground_truth'] = Y
        score = clf.score(X, Y)
        st.write('Model Accuracy is : ',score)
        return df

    def retrain(df):
        data = df
        data = data.drop("url",1)
        data = data.drop("confidence_score",1)
        Y = data['prediction']
        X = data.drop('prediction', axis=1)
        clf_new = LogisticRegression(warm_start=True)
        clf_new.fit(X, Y)
        pickle.dump(clf_new)
        new_score = clf_new.score(X, data['ground_truth'])
        st.write('Model Accuracy is : ',new_score)
        

    button = st.button('Predict')
    if 'button_state' not in st.session_state:
        st.session_state.button_state = False

    if button or st.session_state.button_state:
        st.session_state.button_state = True  

        output_df = predict(df)
        gd = GridOptionsBuilder.from_dataframe(output_df)
        gd.configure_column('prediction',editable=True, type=['numericColumn','numberColumnFilter','customNumericFormat'], precision=0)
        gd.configure_pagination(enabled=True)
        # gd.configure_default_column(groupable=True,min_column_width=1)
        gd.configure_selection(selection_mode='single',use_checkbox=True)
        gridoptions = gd.build()
        grid_table = AgGrid(output_df, gridOptions=gridoptions, reload_data=False, data_return_mode=DataReturnMode.AS_INPUT, update_mode=[GridUpdateMode.SELECTION_CHANGED,GridUpdateMode.MODEL_CHANGED], theme="alpine")
        sel_row = grid_table['selected_rows']
        st.write(sel_row)
        # for i in range(len(grid_table['data'])):
          #   if output_df.loc[i]['prediction'] != grid_table['data']['prediction'][i]:
            #    st.caption(Prediction column data )
        # st.write(grid_table)
        #if st.button("Update Changes"):
        #    new_df = grid_table['data']
        #    new_df.to_csv("data/data.csv", index=False)
        #st.dataframe(pd.DataFrame.from_dict(grid_table))
        if st.button('Save Changes'):
            new_df = grid_table['data']    # overwrite df with revised aggrid data; complete dataset at one go
            new_df.to_csv('data/file1.csv', index=False)  # re/write changed data to CSV if/as required
            
            st.dataframe(new_df)   # confirm changes to df
            if st.button("Retrain"):
                retrain(new_df)

        with st.expander("Don't trust the model predictions? "):
            train_data = df.drop("label",1)
            train_data = train_data.drop("url",1)
            pickle_in = open("LR_model.pkl","rb")
            clf = pickle.load(pickle_in)
            explainer = lime_tabular.LimeTabularExplainer(
                training_data = np.array(train_data),
                feature_names = train_data.columns,
                class_names = ['Benign','Malicious'],
                mode = 'classification'
            )
            exp = explainer.explain_instance(
                data_row = train_data.iloc[1],
                predict_fn = clf.predict_proba
            )
            fig = exp.as_pyplot_figure()
            st.pyplot(fig=fig,)
        

   

    file.close()


if __name__=='__main__':
    st.set_page_config(layout="wide")
    main()