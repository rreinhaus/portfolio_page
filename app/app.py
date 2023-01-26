from turtle import color
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from port_func import one_city_graph, one_postcode_graph,format_city_data,hvplot_map,foliumheatmap, text_toml_results,climate_df,time_predict, denoise_text_and_predict
from port_func import spend_clothing, spend_electricity, spend_food,spend_restaurant, flights,fuel,formatting_plot, job_search, clean_dataframe
from streamlit_folium import folium_static
import folium
import plotly.graph_objects as go
import holoviews as hv
hv.extension("bokeh", "matplotlib")
import holoviews as hv, pandas as pd, colorcet as cc
import pickle


# Growth Rate Function

def growth_per(data):
    growth_rate = np.exp(np.diff(np.log(data['GDP']))) - 1
    growth = [0]
    for grow in growth_rate:
        growth.append(round((grow*100),2))

    return growth


# Layout and Header
st.set_page_config(
            page_title="Rich - Data Services", 
            page_icon="🌌",
            layout="wide", 
            initial_sidebar_state="auto") # collapsed

#  SideBar
st.sidebar.title("Richard's - Data Service")
st.sidebar.markdown("In order to view the work, press on the section you want to view")

# Navigation of the website
navi = st.sidebar.radio('Select a section', ('Home','Data Capture & Formatting', 'Data Vizualizations', 'ML/DL Projects'))

st.sidebar.write("""Please note that work presented here is copyright by Data Rich (datarich@proton.me) and without consent cannot be shared.""")
st.sidebar.write("""May the Force be with you""")


if navi == 'Home':
   
    # Introduction 

    st.markdown('''
    # Welcome!
    This is Data Rich portfolio web application. 
    The website gives information about me as data analyst/scientist and showcases few examples of my capabilites.
    
    The sections include:
    - Data Capture
    - Data Cleaning/Formatting 
    - Data Visualization
    - Machine Learning Algos
    
    Each page has description on how the work was done and what are the results. 
    
    If you like to be in touch for any freelance work email at - datarich@proton.me''')

    st.markdown('---')
    
    # Make sum fun data reporting and maybe sentiment analysis of the elon musk tweets or something else that just looks cool

    # Front Page Visuals
    
elif navi == 'Data Capture & Formatting':
    
    st.markdown('''
    # Climate Change - Carbon Emissions Caculator API
    ### Data Capture
    
    The following fields of questions are connected to Climatiq API, which allows to calculate our carbon emissions from our acitivites.
    
    Tools used:
    - Connecting using requests to Climatiq API
    - Webscraping with BeautifulSoup to get the right airport code on https://www.iata.org/
    - Visualizations - plotly
    ''')
    
    st.markdown('---')
    
    # Food Spending
    
    st.markdown('''#### How much money do you spend per week on food?''')
    
    food_col1, food_col2 = st.columns(2)
    
    with food_col1:
        # Spending amount
        food_amount = st.number_input('Amount that you spend on food')
        
    with food_col2:
        # Currency
        food_currency = st.selectbox('Select a currency for food', ['eur','usd','gbp'])
        
    if food_amount > 0:
        food_emission = spend_food(food_amount,food_currency)
        st.markdown(f'''Your emissions from spending **{int(food_amount)}{food_currency}** on food per week is **{int(food_emission)} kgCO2e per annum**''')
    else:
        food_emission = 0
        st.markdown('''*No amount entered*''')
    
    # Kilometers
    st.markdown('''#### How many km do you drive per week?''')
    
    # km input
    kilometers = st.number_input('Enter number of kilometers you drive')
    
    if kilometers > 0:
        kilometers_emission = fuel(kilometers)
        st.markdown(f'''Your emissions from driving **{int(kilometers)}km** per week is **{int(kilometers_emission)} kgCO2e per annum**''')
    else:
        kilometers_emission = 0
        st.markdown('''*No amount entered*''')
        
    # Clothing
    st.markdown('''#### How much money do you spend per month on clothing?''')
    
    clothing_col1, clothing_col2 = st.columns(2)
    
    with clothing_col1:
        # Spending amount clothing
        clothing_amount = st.number_input('Amount that you spend on clothing')
        
    with clothing_col2:
        # Currency
        clothing_currency = st.selectbox('Select a currency for clothing', ['eur','usd','gbp'])
    
    
    if clothing_amount > 0:
        cloth_emission = spend_clothing(clothing_amount,clothing_currency)
        st.markdown(f'''Your emissions from spending **{int(clothing_amount)}{clothing_currency}** on clothing per month is **{int(cloth_emission)} kgCO2e per annum**''')
    else:
        cloth_emission = 0
        st.markdown('''*No amount entered*''')
    
    # Restaurants
    
    st.markdown('''#### How much money do you spend per month on restaurants?''')
    
    restaurants_col1, restaurants_col2 = st.columns(2)
    
    with restaurants_col1:
        # Spending amount restaurants
        restaurant_amount = st.number_input('Amount that you spend on restaurant')
        
    with restaurants_col2:
        # Currency
        restaurant_currency = st.selectbox('Select a currency for restaurants', ['eur','usd','gbp'])
    
    if restaurant_amount > 0:
        res_emissions = spend_restaurant(restaurant_amount,restaurant_currency)
        st.markdown(f'''Your emissions from spending **{int(restaurant_amount)}{restaurant_currency}** on restaurants per month is **{int(res_emissions)} kgCO2e per annum**''')
    else:
        res_emissions = 0
        st.markdown('''*No amount entered*''')
    
    # Flights 
    st.markdown('''#### How many times per month do you fly and from where to where did you fly most recently?''')
    
    flight_col1, flight_col2,flight_col3 = st.columns(3)
    
    with flight_col1:
        # Spending amount clothing
        flight_amount = st.number_input('Times you are taking a airplane per month')
        
    with flight_col2:
        # From where
        from_flight = st.text_input('Please enter a city you flew from.')
        
    with flight_col3:
        # To where
        to_flight = st.text_input('Please enter a city where you flew to.')
     
    if flight_amount > 0 and from_flight != '' and to_flight != '':
        flight_emissions = flights(flight_amount,from_flight,to_flight)
        st.markdown(f'''Your emissions from flying **{int(flight_amount)}** times per month is **{int(flight_emissions)} kgCO2e per annum**''')
    else:
        flight_emissions = 0
        st.markdown('''*No amount entered*''')
     
    # Electricity   
    st.markdown('''#### How much electricty you spend per month?''')
    # electricity input
    electricty_spent = st.number_input('Enter number of kwh you spend per month')
    
    if electricty_spent > 0:
        electircity_emi = spend_electricity(electricty_spent)
        st.markdown(f'''Your emissions from spending **{int(electricty_spent)}kwh** per month is **{int(electircity_emi)} kgCO2e per annum**''')
    else:
        electircity_emi = 0
        st.markdown('''*No amount entered*''')
        
    
    cli_col1, cli_col2, cli_col3 = st.columns(3)
    
    total_consumption = food_emission + kilometers_emission + flight_emissions + cloth_emission + res_emissions + electircity_emi
    total_consumption = total_consumption/1000
        
    if total_consumption > 0:
        with cli_col2:
            if total_consumption < 7:
                st.markdown(''' You are quite sustainable, keep it on. As you are producing **low carbon (under 7tCO2e)** footprint.''')
            else:
                st.markdown(f'''Your carbon footprint is **{int(total_consumption)}** and it is too big so reduce it to at least **7 tCO2e**''')
            fig_climate = formatting_plot(food=food_emission, fuel=kilometers_emission, 
                                        flights=flight_emissions, clothing=cloth_emission,restaurants=res_emissions,
                                        electricity=electircity_emi)
            st.plotly_chart(fig_climate)
    else:
        pass
    
    
    # Future Work -> Create a way so that some values can be zero. 
    st.write('---')
    
    st.markdown('''
    # Job Aggregator API 
    ### Data Capture - See job openings in 20 different countries for the role you are looking for.
    
    The following fields of questions are connected to Adzuna API, which allows to locate jobs based on your needs in 20 different countries.
    
    Tools used:
    - Connecting using requests to Adzuna API
    - Displaying data in table like format and using f string to perform a detailed GET search.
    ''')
    
    
    st.markdown('---')
    
    occup_col, country_col, contract_col = st.columns(3)
    
    with occup_col:
        occupation = st.text_input('Share an occupation you want to search for')
    
    with country_col:
        country_job = st.selectbox('Select country you want results for', ['at', 'au', 'be', 'br', 'ca', 'ch', 'de', 'es', 'fr', 'gb', 'in', 'it', 'mx', 'nl', 'nz', 'pl', 'ru', 'sg', 'us', 'za'])
    
    with contract_col:
        contract = st.selectbox('Select contract type', ['full_time','part_time','contract','permanent'])
    
    if occupation != '':
        job_search_df = job_search(occupation,contract,country_job)
        st.table(job_search_df)
    
    st.markdown('''---''')
    
    st.markdown(''' 
            # Data Cleaner
            
            The following function allows to upload a dirty csv file and get back a clean one.
            
            The function does the following:
            - Take in a dataframe in CSV format and checks for missing values, if any column contains more than 50 percent of missing values it will remove that column
            - It will fill the missing values with mean or mode depending on the column type
            - It will remove any duplicate rows
            - Check for inconsistencies within the values used in the dataframe and will print the column name which has inconsistencies 
            - It will format the date and numeric columns as per the requirement
            - At the end it will return the user with a clean dataset
            
            ''')
    
    st.markdown('''---''')
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # cleaning
        data = pd.read_csv(uploaded_file)
        clean_df = clean_dataframe(data)
        st.write(clean_df.head())
        
        # Giving option of download
        csv = clean_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
        
        
elif navi == 'Data Vizualizations':
    
    #### 2nd Project - Biodiversity ####
    
    st.markdown('''
    # HousVal Project
    
    Visuals presents UK sold house price locations based on your address input. For instance, it will showcase around what price with your post code other properties were sold and other properties within the city. 
    
    As additional visualization you will be able to see the locations of the highest property prices and regions which were most popular amongst the buyers.''')
    
    st.markdown('---')
    
    graph_data = pd.read_csv('data/HousVal_graph_data_withsalesbucket.csv')
    
    st.write('What is the postcode and city of the property you would like to sell?')

    post_col3,city_col4 = st.columns(2)
    
    with post_col3:
        # post code text field
        post_code = st.text_input('Postcode','NP44 1FN')
    
    with city_col4:
        # city text field
        city_name = st.text_input('City', 'CWMBRAN')
    
    # Full address as one string
    full_address = str(post_code) + ' ' + str(city_name)
    
       # Trigger the valuation
    if st.button(f'Visualizations for the address at {full_address}'):
        
        # Simple insights
        st.write('### Sold property historical data')
        
        display_data = graph_data[graph_data['post_code'] == post_code]
        
        col1, col2, col3,col4 = st.columns(4)
        col1.metric("Number of Properties Found", f"{len(display_data)}")        
        col2.metric("Min Rroperty Sales Price", f"£{display_data.sale_price.min()}")
        col3.metric("Median Property Sales Price", f"£{int(display_data.sale_price.median())}")
        col4.metric("Max Property Sales Price",  f"£{display_data.sale_price.max()}")
        
                    
        viz1, viz2 = st.columns(2)
        
        with viz1:
            # Graph for postcode sale price changes over time
            st.plotly_chart(one_postcode_graph(df=graph_data, post_code= post_code,date_from='2017-01-01',date_to= '2022-12-01'),use_container_width=True)
        
        with viz2:
            # Outlier Free Data
            graph_data = graph_data[(graph_data['sale_price'] >= graph_data['sale_price'].describe()['25%']) & (graph_data['sale_price'] <= graph_data['sale_price'].describe()['75%'])]
            
            # Requires all upper letters
            city_name = city_name.upper()
            
            # Graph for city sale price changes over time
            st.plotly_chart(one_city_graph(df=graph_data,city=city_name,date_from='2021-01-01',date_to= '2022-12-31'),use_container_width=True)


        viz3, viz4 = st.columns(2)
        with viz3:
            # Postcode sales price bucket donut chart
            post_code_df = graph_data[graph_data['post_code'] == post_code]
            
            # Data prep
            data = post_code_df.groupby(pd.qcut(post_code_df["sale_price"],7, duplicates="drop")).agg({"sale_price" : "count"})
            
            final_val = []
            for i in range(len(data)):
                final_val.append(str(data.index[i].left) + '-' + str(data.index[i].right))
                
            data['sale_range'] = final_val
            
            # The donut chart based on price bucket with equal split
            fig_don = go.Figure(data=[go.Pie(labels=data['sale_range'], values=data['sale_price'], hole=.3)])
            fig_don.update_layout(
                            title="Property sale prices grouped into equal buckets",
                            legend_title="Sales Price Buckets",
                            font=dict(
                            color="RebeccaPurple"
                            ))
            st.plotly_chart(fig_don)
            
        with viz4:
            # Top 10 most popular postcodes in the city
            
            # Data
            one_city = format_city_data(graph_data, city_name)
            
            # Viz
            fig = go.Figure(data=[go.Pie(labels=one_city.post_code.value_counts().head(10).keys(), values=one_city.post_code.value_counts().head(10).values, hole=.3)])
            fig.update_layout(
                title="Top 10 most popular postcodes in the city",
                legend_title="Postcodes",
                font=dict(
                color="RebeccaPurple"
                ))
            st.plotly_chart(fig)
            
        viz5, viz6 = st.columns(2)     
            
        with viz5:
            # Sold properties accross the city with specific price range
            
            # Data
            one_city = pd.read_csv('data/plot_test.csv')
            #one_city = second_phaseofformating(one_city)
            
            # Viz
            st.write('Properties across the city within specific sales price ranges')
            st.bokeh_chart(hv.render(hvplot_map(one_city), backend='bokeh'))
            
        with viz6:
            # Heatmap of the properties
            st.write('Locations of the properties within the city')
            st_data = folium_static(foliumheatmap(one_city))
    
    st.markdown('---')
    
    #### 2nd Project - Life Expectancy & GDP ####
    
    st.markdown('''
    # Life Expectancy and GDP Project
    These visualizations represent the comparision of different developed and developing countries in terms of GDP, GDP Growth and Life Expectancy from the period of 2000 - 2015. ''')
 
    df = pd.read_csv('data/all_data_life_expectancy.csv')
    df['Country']= df['Country'].replace('United States of America', 'USA')
    
    df.rename(columns={"Life expectancy at birth (years)": "Life Expectancy"}, inplace=True)
    
    # Each countries dataset
    data_usa = df[df['Country'] == 'USA']
    data_china = df[df['Country'] == 'China']
    data_germany = df[df['Country'] == 'Germany']
    data_chile = df[df['Country'] == 'Chile']
    data_Mexico = df[df['Country'] == 'Mexico']
    data_zimbabwe = df[df['Country'] == 'Zimbabwe']

    # Adding Growth Rate Column for each nation

    data_china.loc[:, ('GDP_Growth')] = growth_per(data_china)
    data_usa.loc[:, ('GDP_Growth')] = growth_per(data_usa)
    data_germany.loc[:, ('GDP_Growth')] = growth_per(data_germany)
    data_chile.loc[:, ('GDP_Growth')] = growth_per(data_chile)
    data_Mexico.loc[:, ('GDP_Growth')] = growth_per(data_Mexico)
    data_zimbabwe.loc[:, ('GDP_Growth')] = growth_per(data_zimbabwe)
    
    frames = [data_usa,data_china,data_germany,data_chile,data_Mexico,data_zimbabwe]
    growth_df = pd.concat(frames)
    countries_list = ['None','USA', 'China', 'Germany', 'Chile', 'Mexico', 'Zimbabwe']

    option = st.selectbox('Select Visualization', ['None','All Countries', 'One vs One', 'One Country'])
    
    #### 2nd Project - Life Expectancy & GDP - Graph Selection ####
    
    if option == 'All Countries':
        gra_col1, gra_col2 = st.columns(2)
        with gra_col1:
            # GDP Average - All
            
            fig1 = px.histogram(df, x="GDP", y="Country",color="Country",histfunc='avg')   
            fig1.update_layout(bargap=0.1, title_text ="Average GDP Per Year Comparision Between Countries", title_x=0.5)     
            st.plotly_chart(fig1,use_container_width=True)

        with gra_col2:
            # Life Expectancy Average - All
            
            fig2 = px.histogram(df, x="Life Expectancy", y="Country",color="Country",histfunc='avg')   
            fig2.update_layout(bargap=0.1, title_text ="Life expectancy at birth (years) Comparision Between Countries", title_x=0.5)     
            st.plotly_chart(fig2,use_container_width=True)
            
        fac_col1,fac_col2 = st.columns(2)
        with fac_col1:  
            # GDP - All
            
            fig3 = px.line(df, x="Year", y="GDP", line_group = "Country", facet_col="Country", facet_col_wrap=3,
                facet_row_spacing=0.07, # default is 0.07 when facet_col_wrap is used
                facet_col_spacing=0.07,color='Country')
            fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig3.update_layout(title_text ="GDP Comparision Between Countries", title_x=0.5)
            fig3.update_yaxes(showticklabels=True,matches=None)
            st.plotly_chart(fig3,use_container_width=True)
        with fac_col2:  
            # Life Expectancy - All
            
            fig4 = px.line(df, x="Year", y="Life Expectancy", line_group = "Country", facet_col="Country", facet_col_wrap=3,
                facet_row_spacing=0.07, # default is 0.07 when facet_col_wrap is used
                facet_col_spacing=0.07,color='Country')
            
            fig4.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            fig4.update_layout(title_text ="Life Expectancy Comparision Between Countries", title_x=0.5)
            fig4.update_yaxes(showticklabels=True,matches=None)
            st.plotly_chart(fig4,use_container_width=True)
            
        # GDP Growth - All  
        
        fig5 = px.line(growth_df, x="Year", y="GDP_Growth", line_group = "Country", facet_col="Country", facet_col_wrap=3,
                facet_row_spacing=0.07, # default is 0.07 when facet_col_wrap is used
                facet_col_spacing=0.07,color='Country')
        
        fig5.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig5.update_layout(title_text ="GDP Growth Comparision Between Countries", title_x=0.5)
        fig5.update_yaxes(showticklabels=True,matches=None)
        st.plotly_chart(fig5,use_container_width=True)
        
    elif option == 'One vs One':
        
        # One vs One Graphs of Life Expectancy and GDP Project
        
        st.markdown("""Select two countries that you would like to compare.""")
        st.markdown('Please bear in mind the country differences, if you select too vastly different countries, it might be hard to analyse due to large GDP difference.')
        country_opt_one = st.selectbox('Select Country 1', countries_list)
        country_opt_two = st.selectbox('Select Country 2', countries_list)
        
        onevone_col1, onevone_col2 = st.columns(2)
        if country_opt_one == 'None' and country_opt_two == 'None':
            st.write('No data to display, please choose a country or two!')
        else:
            with onevone_col1:
                
                # GDP
                
                fig_onevone_1 = px.line(df[(df['Country'] == country_opt_one) | (df['Country'] == country_opt_two)], x="Year", y="GDP", color='Country'
                                        ,markers=True)
                fig_onevone_1.update_layout(title_text =f"GDP comparison between {country_opt_one} & {country_opt_two}", title_x=0.5)
                st.plotly_chart(fig_onevone_1,use_container_width=True)
            
            with onevone_col2:
                # Life Expectancy
                
                fig_onevone_2 = px.line(df[(df['Country'] == country_opt_one) | (df['Country'] == country_opt_two)], x="Year", y="Life Expectancy", color='Country'
                                        ,markers=True)
                fig_onevone_2.update_layout(title_text =f"Life Expectancy comparison between {country_opt_one} & {country_opt_two}", title_x=0.5)
                st.plotly_chart(fig_onevone_2,use_container_width=True)
                    
            # GDP Growth
                
            fig_onevone_3 = px.line(growth_df[(growth_df['Country'] == country_opt_one) | (growth_df['Country'] == country_opt_two)], x="Year", y="GDP_Growth", color='Country'
                                        ,markers=True)
            fig_onevone_3.update_layout(title_text =f"GDP Growth comparison in % between {country_opt_one} & {country_opt_two}", title_x=0.5)
            st.plotly_chart(fig_onevone_3,use_container_width=True)
    
    elif option == 'One Country':
        st.write('The main idea behind this section is to get more detailed of one specific country')
        country_option = st.selectbox('Select a Country', countries_list)
        
        if country_option == 'None':
            st.write('No country to display! Select a country, please!')
        else:
            one_col1, one_col2 = st.columns(2)
            with one_col1:
                # GDP
                
                fig_one_1 = px.line(df[df['Country'] == country_option], x="Year", y="GDP", color='Country'
                                                ,markers=True)
                fig_one_1.update_layout(title_text =f"{country_option} - GDP in trillions of USD dollars", title_x=0.5)
                st.plotly_chart(fig_one_1,use_container_width=True)
            
            with one_col2:
                # Life Expectancy
                
                fig_one_2 = px.line(df[df['Country'] == country_option], x="Year", y="Life Expectancy", color='Country'
                                                ,markers=True)
                fig_one_2.update_layout(title_text =f"{country_option} - life expectancy", title_x=0.5)
                st.plotly_chart(fig_one_2,use_container_width=True)
                    
            one_col3, one_col4 = st.columns(2)
            
            with one_col3:
                # GDP Growth
                
                fig_one_3 = px.line(growth_df[growth_df['Country'] == country_option], x="Year", y="GDP_Growth", color='Country'
                                            ,markers=True)
                fig_one_3.update_layout(title_text =f"{country_option} - GDP growth comparison in %", title_x=0.5)
                st.plotly_chart(fig_one_3,use_container_width=True)
            
            with one_col4:
                # GDP versus Life Expectancy
                fig_one_4 = px.line(df[df['Country'] == country_option], x="GDP", y="Life Expectancy", color='Country'
                                                ,markers=True, text='Year')
                fig_one_4.update_layout(title_text =f"{country_option} - GDP versus life expectancy ", title_x=0.5)
                st.plotly_chart(fig_one_4,use_container_width=True)
    
    st.markdown('---')
    
    #### 3rd Project - Biodiversity ####
    
    st.markdown('''
    # Biodiversity Project
    The following visuals will represent most seen species and specie endangerment statuses across for 4 different USA National Parks.''')
 
    # Data Prep for Second Project
    
    data_observations = pd.read_csv('data/observations_biodiversity.csv')
    data_species = pd.read_csv('data/species_info_biodiversity.csv')
    total_df_biodiversity = pd.read_csv('data/total_df_biodiversity.csv')
    risk_df_biodiversity = pd.read_csv('data/risk_df_biodiversity.csv')
    
    def in_risk_zone(x):
        if x != 'Safe':
            x = 'In Risk'
        elif x == 'In Recovery':
            x = 'Safe'

        return x
    
    def top5most(data, park):
        more_risk_df = data[data['park_name'] == park]
        risk_view = more_risk_df[['observations', 'scientific_name','park_name']].groupby('scientific_name').sum()
        top5_most = risk_view.sort_values(by='observations').head(5)
        return top5_most

    data_species['Cleaned_Status'] = data_species['conservation_status'].apply(in_risk_zone)
    plot_table_bio = pd.crosstab(data_species['conservation_status'][data_species['conservation_status'] != 'Safe'],
                             data_species['category'][data_species['conservation_status'] != 'Safe'])
    plot_table_bio['conservation_status_clean'] = ['Endangered', 'In Recovery', 'Species of Concern', 'Threatened']
    
    # Category Selection - Biodiversity
    
    option_bio = st.selectbox('Select Visualization', ['None','Conservation Status', 'Most Popular Species'])
    
    # Visualizations  - Biodiversity
    
    if option_bio == 'Conservation Status':
        # Table - Category & Conservation Status
        
        st.write('Below are presented how many each animal or plant category have endangered species.')
        st.table(plot_table_bio[['Amphibian', 'Bird', 'Fish', 'Mammal', 'Nonvascular Plant', 'Reptile',
       'Vascular Plant']])
        bio_col1, bio_col2 = st.columns(2)
        
        with bio_col1:
            # Stacked Bar Plot - Status & Category & Observations
    
            fig1_bio = px.bar(plot_table_bio, x="conservation_status_clean", y=['Amphibian', 'Bird', 'Fish', 'Mammal', 'Nonvascular Plant', 'Reptile',
       'Vascular Plant'])   
            fig1_bio.update_layout(bargap=0.1, title_text ="Specie categories by their endangerement status", title_x=0.5,legend_title_text='Categories')     
            st.plotly_chart(fig1_bio,use_container_width=True)
        
        with bio_col2:
            # Top 5 Most Endangered Species by Park
            
            park_option = st.selectbox('Select Park To Display 5 Most Endangered Species In It', ['None','Yosemite National Park', 'Great Smoky Mountains National Park', 'Yellowstone National Park','Bryce National Park'])
            st.bar_chart(top5most(risk_df_biodiversity,park_option))            
        
    if option_bio == 'Most Popular Species':
        # Most Popular Specie by Park
        park_option_two = st.selectbox('Select Park To Display 5 most popular species in it', ['None','Yosemite National Park', 'Great Smoky Mountains National Park', 'Yellowstone National Park','Bryce National Park'])
        st.bar_chart(data_observations[data_observations['park_name'] == park_option_two].groupby(['scientific_name']).sum().sort_values(by='observations',ascending=False).head(5))
        
        # Comparision between parks and their most popular species
        fig2_bio = px.bar(total_df_biodiversity,x='scientific_name',y='Observations',color='Park',barmode="group")
        fig2_bio.update_layout(bargap=0.1, title_text ="Comparision between parks and their most popular species", title_x=0.5)     
        st.plotly_chart(fig2_bio,use_container_width=True)  
    
elif navi == 'ML/DL Projects':
   
    # Intro to HousVal
    st.markdown('''
    # HousVal
    
    Goal: predict sales price of a house, just based on the location (only in the UK)
    
    Overview:
    
    Within this project, I have used UK Land Registry public sample data about the properties sold within the last 10-15 years to predict the house price based on its location. 
    In order to create the ML model and train it on at least 40k data points, I performed the following:
    
    - Exploratory data analysis for categorical and numerical values.
    - Coverted property full addresses into geocoordinates using Nominatim.
    - Performed feature reduction based on feature importance.
    - Scaled the numerical values and performed one-hot-encoding for the cateogrical variables. 
    - Performed cross-validation to fine tune and try out different ML models (Decisiontreeregressor, XGBoostregressor, SGDRegressor and RandomForestRegressor).
    - Decided the best model based on R2 score (which in this case was RandomForestRegressor and score being 0.5)

    In addition to the ML model, you will see different insights based on the property location, for instance what was the price of properties sold with the same postcode.    
    
    ''')
    
    st.markdown('---')     
    
    graph_data = pd.read_csv('data/HousVal_graph_data_withsalesbucket.csv')
    
    st.write('What is the address of the property you would like to sell?')

    number_col1, street_col2, post_col3,city_col4 = st.columns(4)
    
    with number_col1:
        # street number text field
        street_num = st.text_input('Street Number','35')

    with street_col2:
        # Street name text field
        street_name = st.text_input('Street Name', 'cremer street')
    with post_col3:
        # post code text field
        post_code = st.text_input('Postcode','E1 8HD')
    
    with city_col4:
        # city text field
        city_name = st.text_input('City','london')
    
    # Full address as one string
    full_address = str(street_num) + ' ' + str(street_name) + ' ' + str(post_code) + ' ' + str(city_name)
    
    # Trigger the valuation
    if st.button(f'Value property at {full_address}'):
    
        # Open the file in binary mode
        with open('ml_models/40kdata_final_model.sav', 'rb') as file:
            
            # Call load method to deserialze
            model = pickle.load(file)
            
        # ML model prediction
        value = text_toml_results(full_address=full_address,model=model)
        
        st.write(f'### The provided address property value is estimated to be - *£{value}*')   
        
        # Status to showcase when it has run succesfully and when it has failed

    st.markdown('''
    # Temperature Change Predictor
    
    - Dataset from - https://www.kaggle.com/datasets/sevgisarac/temperature-change''')
    
    area_text, start_date_num, finish_date_num = st.columns(3)
    
    with area_text:
        # Area of temperature prediction
        area_string = st.text_input('Country To Predict')

    with start_date_num:
        # Street name text field
        start_date_int = st.text_input('Starting Year of Prediction')
        
    with finish_date_num:
        # post code text field
        finish_date_int = st.text_input('End Year of Prediction')
    
    # fix the button
    predict_temp_but = st.button('Press to predict')
    if predict_temp_but:
        clim_df = climate_df('Latvia')
        if len(clim_df) == 31:
            st.write('This area is not supported currently.')
        else:
            st.plotly_chart(time_predict(clim_df,int(start_date_int),int(finish_date_int),area_string),use_container_width=True) 
        
    ######### News DL Project #########
    
    st.markdown('''
                # Fake news predictor
                
                - Deep Learning NLP model that predicts whether given text is fake or valid. 
                - Glov model with 99 percent accuracy.
                ''')
    
    # Getting the text
    text_touse = st.text_input('Please enter or copy the text you want to check')
    
    if st.button(f'Start the prediction (only after text has been inputted)'):
        
        # Predicting the news
        news_prediction = denoise_text_and_predict(text_touse)
        st.write(news_prediction)
