
# Front-End 
from re import template
from numpy.core.numeric import _full_like_dispatcher
import streamlit as st
import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import base64
from iso3166 import countries as codes
import hydralit_components as hc
from streamlit import caching




# defaults
plt.style.use('seaborn-whitegrid')



# GLOBALS
mapping = {"Côte D'Ivoire": "Côte d'Ivoire",
 'Syria': 'Syrian Arab Republic',
 'United States': 'United States of America',
 'Tanzania': 'Tanzania, United Republic of',
 'Vietnam': 'Viet Nam',
 'Congo (Democratic Republic Of The)': 'Congo',
 'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
 'Iran': 'Iran, Islamic Republic of',
 'Russia': 'Russian Federation',
 'Burma': 'Myanmar',
 'South Korea': 'Korea, Republic of',
 'Taiwan': 'Taiwan, Province of China'}




# FONTS 
font_url0 = ''' <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Kaisei+HarunoUmi:wght@700&display=swap" rel="stylesheet">
'''
font_url1 = '''<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,800&display=swap" rel="stylesheet">
'''

fontname0 = "Kaisei HarunoUmi"
fontname1 = "Playfair Display"




def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded




@st.cache(persist=True, allow_output_mutation=True)
def load_data(about=True):

    if not about:
        dmajcity = pd.read_csv("./data/AvgTempMajorCity.csv").drop('Unnamed: 0',axis=1)
        
        return dmajcity
    else:
        coords = pd.read_csv('./data/choropleth-dat.csv').drop('Unnamed: 0',axis=1)
        return coords



@st.cache(persist=True, allow_output_mutation=True)
def load_provinces():

    # dcity = pd.read_csv("./data/unis-country.csv").drop('Unnamed: 0',axis=1)
    india = pd.read_csv("./data/india.csv").drop('Unnamed: 0',axis=1)
    us = pd.read_csv("./data/USA.csv").drop('Unnamed: 0',axis=1)
    uk = pd.read_csv("./data/UK.csv").drop('Unnamed: 0',axis=1)

    dcity = pd.concat([india,us,uk])
    unis = pd.read_csv("./data/unis.csv").drop('Unnamed: 0',axis=1)

    return dcity, unis
    # return unis

@st.cache(persist=True)
def median_agg(data, sequence='Month',loc='Country',countryname=None,returnlists=False):
      
    # '''
    # Median Aggregation of AverageTemperatures by Country
    # '''
    if loc == 'Country':    
        df = pd.DataFrame(data.groupby([loc, sequence])[['AverageTemperature','AverageTemperatureUncertainty']].median())
        if countryname !=None:
            try:
                return df.loc[countryname, :]
            except:
                print('Invalid Country Name')
        else:
            return pd.DataFrame(data.groupby([loc, sequence])[['AverageTemperature','AverageTemperatureUncertainty']].median())
    elif loc == 'City':
        if countryname != None:
            df = pd.DataFrame(data.groupby(['Country',loc, sequence])[['AverageTemperature','AverageTemperatureUncertainty']].median())
            try:
                return df.loc[countryname, :]
            except:
                print('Invalid Country Name')

        else:
            return pd.DataFrame(data.groupby([loc, sequence])[['AverageTemperature','AverageTemperatureUncertainty']].median())
            
    else:
        return None
    

def order_monthidx(df):
    cats = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sept', 'Oct', 'Nov', 'Dec']
    df.index = pd.CategoricalIndex(df.index, categories=cats, ordered=True)
    df = df.sort_index()
    return df

# Seasonal Pattern
def fetch_seasonal_pattern(ex):
    
    cats = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sept', 'Oct', 'Nov', 'Dec']
    ex = ex.reset_index()
    ex['Month'] =  pd.CategoricalIndex(ex['Month'], categories=cats, ordered=True)
    seasons = list(ex.sort_values(by='Month')['Seasons'].unique())
    ex = ex.sort_values(by='Month')
    return seasons, ex


@st.cache(persist=True)
def city_synopsis(data, city):
    
    # Dictionary of everything - CITY : COMPARATIVE SUMMARY
    package = {}

    df = data[data['City']==city]
    
    # Lowest Ever
    least_series = df.iloc[df['AverageTemperature'].argmin(),:]
    
    min_avgT = least_series['AverageTemperature']
    min_season = least_series['Seasons']
    min_month = least_series['Month']
    min_year = least_series['Year']
    min_deltaT = least_series['AverageTemperatureUncertainty']
    
    # Highest Ever
    high_series = df.iloc[df['AverageTemperature'].argmax(),:]
    max_avgT = high_series['AverageTemperature']
    max_season = high_series['Seasons']
    max_month = high_series['Month']
    max_year = high_series['Year']
    max_deltaT = high_series['AverageTemperatureUncertainty']
    
    maxdata = {'avgT':max_avgT, 'season':max_season, 'deltaT':max_deltaT,
                  'year':max_year, 'month':max_month}
    mindata = {'avgT':min_avgT, 'season':min_season, 'deltaT':min_deltaT,
                  'year':min_year, 'month':min_month}
    
    package['Least'] = mindata
    package['Highest'] = maxdata

    
    summarydf = df.groupby(['Month','Seasons'])[['AverageTemperature','AverageTemperatureUncertainty']].mean()

    
    # hemisphere
    hemisphere = df['Hemisphere'].values[0]
    package['hemisphere'] = hemisphere
    
    #seasons
    seasons, summary_df = fetch_seasonal_pattern(summarydf)
    

    seasonal_avg = {}
    for season in seasons:
        temp = summary_df[summary_df['Seasons']==season][['AverageTemperature','AverageTemperatureUncertainty']].mean()
        seasonal_avg[season] = list(temp.values)
        
    package['seasons'] = seasonal_avg
    

    
    return package



def summary_markdown(city, data, diff, dcity, country):

    st.markdown(f'<h1 style="font-family:{fontname1}; text-decoration:underline; font-size:30px; text-align:center; padding-right:70px">{city}</h1>', unsafe_allow_html=True)

    st.markdown('')
    st.markdown('')

    seasons = list(data['seasons'].keys())
    st.subheader("**Seasonal pattern**")
    st.markdown("<span style='color:royalblue;'>***{}***</span>".format("→".join(seasons)),unsafe_allow_html=True)

    st.subheader('**Seasonal Average Temperatures**')
    for season, templist in zip(data['seasons'].keys(),data['seasons'].values()):
        
        avgT, deltaT = templist
        st.metric(label=season, value=str("%.2f" % avgT)+' °C', delta='ΔT '+str("%.2f" % deltaT)+' °C',delta_color='inverse')

    if diff:
        for x in range(6):
            st.markdown('')

    # seperator 
    st.markdown('***')

    st.subheader('**Extreme temperatures ever recorded**')
    st.markdown('')
    st.markdown('')

    least = data['Least']
    highest = data['Highest']
    

    # least 
    st.markdown(f"<span style='font-size:23px; font-weight:bold; font-family:{fontname0}'> → In the Year - {least['year']}, {least['month']} ({least['season']})</span>",unsafe_allow_html=True)
    st.metric(label='Lowest Ever', value=str("%.2f" % least['avgT'])+' °C', delta='ΔT '+str("%.2f" % least['deltaT'])+' °C',delta_color='inverse')
    
    # highest
    st.markdown(f"<span style='font-size:23px; font-weight:bold; font-family:{fontname0}'> → In the Year - {highest['year']}, {highest['month']} ({highest['season']})</span>",unsafe_allow_html=True)
    st.metric(label='Highest Ever', value=str("%.2f" % highest['avgT'])+' °C', delta='ΔT '+str("%.2f" % highest['deltaT'])+' °C',delta_color='inverse')
    


    # seperator 
    st.markdown('***')

    # Box plot 
    st.subheader('**Year Average Temperature Distribution**')
    monthly = median_agg(dcity,loc='City',countryname=country).loc[city,:]

    x_vals = monthly['AverageTemperature']

    
    fig = go.Figure()
    fig.add_trace(go.Box(y=x_vals, name=city, marker_color = 'dodgerblue'))

    fig.update_layout(
        yaxis_title="Temperature",
        font=dict(
            family="sora",
            size=12,
        )
    )
    
    st.plotly_chart(fig,use_container_width=True)



    



def plottimeseries(data,cityname, attr='AverageTemperature',loc='City',multiple=None):
    
    ## Prompt 
    if attr == 'AverageTemperature':
        prompt = 'Average Temperature'
    else:
        prompt = 'Average Temperature Uncertainty (Δ T)'
    
    if loc=='City':

        if 'Month' in data.reset_index().columns:
            plt.figure(figsize=(15,8))
            
            if multiple == None:
                data = data.xs(cityname, level=0, drop_level=True)
                data = order_monthidx(data)
                x = data.index
                y = data[attr].values
                if attr == 'AverageTemperature':
                    plots = sns.barplot(x=x,y=y,palette='Blues_r',label=cityname,edgecolor='black')

                    for bar in plots.patches:
                        
                        plots.annotate(format(bar.get_height(), '.2f'),
                                    (bar.get_x() + bar.get_width() / 2,
                                        bar.get_height()), ha='center', va='center',
                                    size=18, xytext=(0, 8),
                                    textcoords='offset points')
                    
                    plt.title('Monthly {} Recorded in {}'.format(prompt,cityname),size=25)
                    plt.ylabel('Temperatures °C',size=23)
                    plt.xlabel('Months',size=23)
                    plt.xticks(size=15)
                    plt.yticks(size=15)
                else:
                    sns.lineplot(x=x,y=y,color='royalblue',marker='o')
                    plt.title('Monthly {} Recorded in {}'.format(prompt,cityname),size=25)
                    plt.ylabel('Temperatures °C',size=23)
                    plt.xlabel('Months',size=23)
                    plt.xticks(size=15)
                    plt.yticks(size=15)
                # plt.legend(bbox_to_anchor=(1.13, 1.05))
                
            
            else:
                coords = []
                if len(multiple) <=3 and len(multiple) <=5:
                    for cityname in multiple:
                        df = data.xs(cityname, level=0, drop_level=True)
                        df = order_monthidx(df)
                        x = df.index
                        y = df[attr].values
                        coords.append((x,y))
                

                # setting plot attributes
                plotattrs = {}
                colors = ['crimson','magenta','darkorange','darkslateblue']
                for cityname in multiple[1:]:
                    plotattrs[cityname] = {'marker':'o', 'color':colors.pop(0)}
                
                count = 0
                for coord,cityname in zip(coords,multiple):
                    x,y = coord

                    if count < 1:
                        sns.barplot(x=x,y=y,palette='Blues_r',label=cityname, edgecolor='black')
                        count +=1
                    else:
                        pltattrs = plotattrs[cityname]
                        sns.lineplot(x=x,y=y,color=pltattrs['color'],marker=pltattrs['marker'],label=cityname)
                    plt.title('Monthly {} Recorded in {}'.format(prompt,cityname),size=25)
                    plt.ylabel('Temperatures °C',size=23)
                    plt.xlabel('Months',size=23)
                    plt.xticks(size=15)
                    plt.yticks(size=15)
                    plt.legend(bbox_to_anchor=(1.13, 1.05))
                    
                
            
        else:
            data = data.xs(cityname, level=0, drop_level=True)
            plt.figure(figsize=(15,8))
            x = data.index
            y = data[attr].values
            sns.lineplot(x=x,y=y,label=cityname,marker='P',color='royalblue',palette='inferno')
            plt.title('Yearly {} Recorded in {}'.format(prompt,cityname),size=25)
            plt.ylabel('Temperatures °C',size=23)
            plt.xlabel('Years',size=23)
            plt.xticks(size=15)
            plt.yticks(size=15)
            # plt.legend(bbox_to_anchor=(1.13, 1.05))
            

    else:
        pass


def plotlychart(df,attr='AverageTemperature',cityname=None,sequence='Monthly'):
    

    # pio.templates.default = 'plotly_white'
        ## Prompt 
    if attr == 'AverageTemperature':
        prompt = 'Average Temperature'
    else:
        prompt = 'Average Temperature Uncertainty (Δ T)'
    
    # Template
    pio.templates.default = 'plotly_white'
    

    if sequence == 'Monthly':

        if not attr == 'AverageTemperatureUncertainty':

            x = list(df.index)
            y0 = np.round(df[attr].values,3)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=x, y=y0, text=y0, textposition='outside',
                                marker={'color': y0, 'colorscale': 'Blues'},name='Average'))
            # fig.add_trace(go.Bar(x=x, y=y1,
            #                 marker_color='crimson',name='deltaT'))

            
            fig.update_layout(font=dict(size=11,family='sora'),template='ggplot2+seaborn' ,title='{} {} Recorded in {}'.format(sequence,prompt,cityname), 
                            xaxis_title='Months',
                            yaxis_title='Temperature C',
                            title_x=0.5
                            )

        else:

            x = list(df.index)
            y0 = np.round(df[attr].values,3)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y0,name='Average',mode='lines+markers',
                                    line=dict(color='dodgerblue',width=2)))
            # fig.add_trace(go.Bar(x=x, y=y1,
            #                 marker_color='crimson',name='deltaT'))


            fig.update_layout(font=dict(size=11,family='sora'), template='ggplot2+plotly_dark', title='{} {} Recorded in {}'.format(sequence,prompt,cityname), 
                            xaxis_title='Months',
                            yaxis_title='Temperature C',
                            title_x=0.5
                            )

        

    elif sequence == 'Yearly':
        x = list(df.index)
        y0 = df[attr]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y0,name='Average',mode='lines',
                                line=dict(color='royalblue',width=2)))
        # fig.add_trace(go.Bar(x=x, y=y1,
        #                 marker_color='crimson',name='deltaT'))


        fig.update_layout(font=dict(size=11,family='sora'), template='ggplot2+plotly_dark', title='{} {} Recorded in {}'.format(sequence,prompt,cityname), 
                          xaxis_title='Months',
                          yaxis_title='Temperature C',
                          title_x=0.5
                         )


    return fig


# For Dash Board Section -- HTML unordered list generator 
def sections_cs(section, subsection=False):
    
    if not subsection:
        id = '-'.join(section.lower().split())
        space= ''
        html_string = f'* <a href="#{id}"><span style="font-family: {fontname0}; color:black; font-size:18px; ">{section.capitalize()} <br>{space}</span></a>'
        return html_string, id

    else:
        pattern = []

        id = '-'.join(section.lower().split())
        # html_string = f'* <a href="#{id}"><span style="font-family: {fontname0}; color:black; font-size:18px; ">{section.capitalize()} <br>{space}</span></a>',
        if isinstance(subsection, list):
            for s in subsection:
                pattern.append('''<li><p style="font-size:15px;">{}</p></li>'''.format(s))
                
                
            pattern = ''.join(pattern)
            html_string = '''* <a href="#{}"><span style="font-family: {}; color:black; font-size:18px; ">{} <br> </span></a><ul>{}</ul>'''.format(id,fontname0,section.capitalize(),pattern)
            
            return html_string, id
            
    
        return None



def about_cs():

    df = load_data(about=True)


    fig = go.Figure(data=go.Choropleth(
        locations = df['CODE'],
        z = df['AvgTemp'],
        text = df['COUNTRY'],
        colorscale = 'RdBu_r',
        autocolorscale=False,
        marker_line_color='black',
        marker_line_width=0.9,
        colorbar_ticksuffix = '°C',
        colorbar_title = 'Temperature °C',
    ))

    fig.update_layout( geo=dict( showframe=True, showcoastlines=True, projection_type='natural earth'),font=dict(size=15,family='sora'),template='plotly_white' ,height=600,width=900,dragmode=False)
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)


    about_info = '''
    Forecasting the Long-term global warming rate is significant in various fields, including climate research, agriculture, energy, medicine, and many more. This is project showcases a detailed analysis of the change in temperatures for different countries. Challenge dwells on creating a reliable, efficient, statistically reliable model on extensive data set and accurately capture the relationship between average annual temperature. This data will be used as the foundational information to train multiple Machine Learning models with approaches like Deep Neural Networks and LSTMs, Time series forecasting algorithms like FbProphet, and fine-tune the best functional approach out of these approaches to get accurate generalizations. Obtaining this analyzed and forecasted data ahead of time allows the use of long-term mitigation methods.
    '''

    st.markdown(f"""<p style="font-family:'georgia'; font-size:17px; text-align:justify;"><b>Abstract:</b>{about_info}</p>""", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:27px; text-align:center;">Choropleth Projection of Average Temperatures Y.1743-2013</h1>', unsafe_allow_html=True)
    cols = st.columns([1,6,1])
    cols[1].plotly_chart(fig)

    st.markdown('---')
    

def traveller_cs(data, mig):

    # info = {"Migrating": 'Migration: for students who are moving out of their home country for education & For people starting their professional endeavours in abroad.',
    #         "Visiting": "Visiting: for people who are travelling to a Country and its major cities."}
    info = {"Migrating": 'Migration: for students who are moving out of their home country for education & For people starting their professional endeavours in abroad.',}
            

    # instantiation
    hcity, mcity = None, None


    clist = list(np.sort(data['Country'].unique()))
    countries = ['Select Country', *clist]

    # data Access
    dcity, unis = mig
    

    scols = st.columns([6,3])
    category = scols[0].selectbox('What type are you?', ['Migrating','Visiting'],key='category')
    st.markdown(f"""> <p style="font-family:'georgia'; font-size:17px; text-align:justify;">{info[category]}</p>""", unsafe_allow_html=True)
    st.markdown('***')



    if category == 'Migrating':

        st.markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:37px; text-align:center;">{category}<br></h1>', unsafe_allow_html=True)
        st.markdown('')

        # A predefined list of purpose of migration 
        purpose = ['Masters/PhD', 'Employment & Others']
        motive = scols[1].radio('Motive',purpose,key='purpose')


        if motive == 'Employment & Others':

    
            frm_cols = st.columns(2)
            
            with frm_cols[0]:
                
                hcountry = st.selectbox('Home Country',countries,key='hcountries')
                cltemp = data[data['Country']==hcountry]['City'].unique()
                hcitylist = ['Select City', *cltemp]


                mcountry = st.selectbox('Migrating to?',countries,key='mcountries')
                cltemp = data[data['Country']==mcountry]['City'].unique()
                mcitylist = ['Select City', *cltemp]
                
                st.markdown('')
                
                

            with frm_cols[1]:
                
                with st.form(key='emp'):
                    hcity = st.selectbox('Home City',hcitylist,key='hcities')
                    mcity = st.selectbox('City',mcitylist,key='mcities')

                    submit_button = st.form_submit_button(label='Analyse 🧐')


                    
                    
        
        elif motive == 'Masters/PhD':


            clist = np.sort(dcity['Country'].unique())
            countries = ['Select Country', *clist]

            ulist = unis['uni-country'].unique()
            universities = ['Select University', 'Other', *ulist]
            
            hcol = st.columns([6,3])

            hcountry = hcol[0].selectbox('Home Country', countries, key='masters-hc')
            uni = hcol[0].selectbox('University?',universities,key='unis')

            hcol[0].markdown("*Please be aware of the city/branch of the university you are applying to, if your city doesn't exist in the list, try with the cities nearby for general intuition.*")

            if not hcountry == 'Select Country' and not uni == 'Select University':
                tempcity = dcity[dcity['Country']==hcountry]['City'].unique()
                try:
                    uni,code = uni.split(',')
                except:
                    uni,_,code = uni.split(',')
                mcountry = unis[unis['codes'] == code]['Country'].unique()[0]
                ucities = dcity[dcity['Country']==mcountry]['City'].unique()
                
                
                hcities = ['Select City', *tempcity]
                with hcol[1].form(key='Cities'):
                    hcity = st.selectbox('Home City', hcities, key='masters-hcity')
                    mcity = st.selectbox('Branch/City', ucities, key='uni-branches')

                    submit_button = st.form_submit_button(label='Analyse 🧐')
                    
                
        

    
        
    if not hcity == None and not mcity == None:

        if not hcity == 'Select City' and  not mcity == 'Select City':
            # checks 
            if hcity == mcity: 
                st.error("*Home City and the city you're Migrating to should'nt be the same, if it is then this analysis doesn't make sense.*")
        

            else:
                st.markdown('---')

                # Fetch Dictionaries from the pipeline

                homedata = city_synopsis(dcity, hcity)
                awaydata = city_synopsis(dcity, mcity)


                home, space, away = st.columns([6,0.5,6])


                y0 = np.random.randn(50)

                fig = go.Figure()
                fig.add_trace(go.Box(x=y0, name='Sample A',
                                marker_color = 'indianred'))
                            


                with home:

                    # For Spacing & organisation after seasonal averages
                    if len(homedata['seasons'].keys()) < len(awaydata['seasons'].keys()):
                        diff = True
                    else:
                        diff = False
                    
                    summary_markdown(hcity, homedata, diff, dcity, hcountry )


                    
                    

                with space:
                    for x in range(40):
                        st.markdown('|')



                with away:

                    # For Spacing & organisation after seasonal averages
                    if len(awaydata['seasons'].keys()) < len(homedata['seasons'].keys()):
                        diff = True
                    else:
                        diff = False
                    
                    summary_markdown(mcity, awaydata, diff, dcity, mcountry)
                    
                st.markdown('***')
                





def analytics_cs(data):
        
    clist = list(np.sort(data['Country'].unique()))

    countries = ['Select Country', *clist]

    slcols = st.columns([3,2])

    countryname = slcols[0].selectbox('Country',countries,key='country')
    emslcols = slcols[1].empty()

    st.markdown('***')

    if not mapping.get(countryname) == None:
        countryname = mapping.get(countryname)


       

    if not countryname == 'Select Country':

        monthex = median_agg(data,sequence='Month',loc='City',countryname=countryname)
        yearex = median_agg(data,sequence='Year',loc='City',countryname=countryname)
        

        try:
            if monthex == None:
                orgc = list(mapping.keys())[list(mapping.values()).index(countryname)]
                monthex = median_agg(data,sequence='Month',loc='City',countryname=orgc)
                yearex = median_agg(data,sequence='Year',loc='City',countryname=orgc)

        except:
            pass


        # st.write(orgc)
        cityname = emslcols.selectbox('Major Cities',pd.Series(dict(monthex.index)).index,key='cities')
        
       
        cols = st.columns(2)
        try:
            
            cols[0].markdown(f'''<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:37px; ">{countryname} <img src="https://flagcdn.com/256x192/{codes.get(countryname).alpha2.lower()}.png" width="96" height="75" alt="{countryname}"><p style="text-align:left; font-weight:bold;">{cityname}</p></h1>''', unsafe_allow_html=True)
            countryname = orgc
        except:
            
            pass
            # cols[0].markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:47px; ">{orgc} <img src="https://flagcdn.com/256x192/{codes.get(orgc).alpha2.lower()}.png" width="96" height="75" alt="{countryname}"><p style="text-align:left; font-weight:bold;">{cityname}</p></h1>', unsafe_allow_html=True)
            # countryname = orgc
        
        
        
        
        # Alignment
        st.markdown('')
        st.markdown('')
        # cols[1].markdown('')



        # Sections

        
        secs = cols[1].expander('Dashboard Sections',True)

        htmlstr0, ts_id = sections_cs('Time Series',['Selective Time-Series'])
        secs.markdown(htmlstr0, unsafe_allow_html=True)

        htmlstr1, qs_id = sections_cs('Quantified Summary')
        secs.markdown(htmlstr1, unsafe_allow_html=True)

        htmlstr2, ca_id = sections_cs('Comparative Analysis')
        secs.markdown(htmlstr2, unsafe_allow_html=True)
        
        htmlstr3, cs_id = sections_cs('Contrast & Summary')
        secs.markdown(htmlstr3, unsafe_allow_html=True)

        # Ids
        # st.write([ts_id, qs_id, ca_id, cs_id])

        
        
        st.markdown("<span id='timeseries'></span>",unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:37px; text-align:center;">Time Series</h1>', unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')


        



        
        # CONTRFOL 
        if not countryname=='Select Country':
                # # Key Notes can Include the final machine learning quantified Delta T with st.metric and arrow to indicate an increase or a decrease
                # expcol = cols[1].expander(f'key-notes of {countryname}',False)
                # expcol.code(list(dict(monthex.index).keys()))

                
            # Plots 
            pltcols = st.columns(2)
            
            fig1 = plotlychart(monthex.loc[cityname,:], cityname=cityname, sequence='Monthly', attr='AverageTemperatureUncertainty')
            pltcols[0].markdown('')
            pltcols[0].markdown('')
            # Montly Average Bar Chart 
            pltcols[0].pyplot(plottimeseries(monthex,cityname))
            # Monthly Average Delta T
            pltcols[1].plotly_chart(fig1)
            

            # line break
            st.markdown('')
            
        
            tscols = st.columns([5,0.2,5])
            ts_avgtemp = plotlychart(yearex.loc[cityname,:], cityname=cityname, sequence='Yearly', attr='AverageTemperature')
            ts_avgun = plotlychart(yearex.loc[cityname,:], cityname=cityname, sequence='Yearly', attr='AverageTemperatureUncertainty')
            
            tscols[0].plotly_chart(ts_avgtemp)
            tscols[2].plotly_chart(ts_avgun)

                



            

        st.markdown("<span id='quantified-summary'></span>",unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:37px; text-align:center;"  >Quantified Summary</h1>', unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')





        # Dont Delete -- Redirection
        st.markdown(f'<a href="#top"><span style="font-family: {fontname0}; color:darkblue; font-size:15px; padding-left:1090px; ">Jump to top of page</span></a>',unsafe_allow_html=True)


@st.cache(persist=True)
def prophetize(dmajcity, province):
    
    df = dmajcity[dmajcity['City']==province]
    
    
    # initialise model
    model = Prophet(yearly_seasonality=True)
    
    dt_prophet = df[['dt','AverageTemperature']]
    dt_prophet.columns = ['ds','y']
    
    # fit model
    model.fit(dt_prophet)
    
    future_dates=model.make_future_dataframe(periods=2920)
    prediction=model.predict(future_dates)
    pred_df = prediction
    pred_df['year'] = pred_df['ds'].dt.year
    pred_df['day'] = pred_df['ds'].dt.day
    pred_df['month'] = pred_df['ds'].dt.strftime("%b")
    
    # prune dates
    pred_df = pred_df[pred_df['year']>2013]
    pred_df = pred_df[pred_df['day']==1]
    pred_df = pred_df.drop('day',axis=1)
    
    return pred_df



def project_yhat(data,city,year,mode='compound'):
    
    
    # read data
    
    hydorg = pd.read_excel('./data/HydAvg18-20Org.xlsx')
    tororg = pd.read_excel('./data/TorontoAvg18-20Org.xlsx')

    if mode == 'compound':
        data = data[(data['year']>=2018)&(data['year']<2021)]
        prompt = 'Prophet Forecast for Average Temperatures in {}'
        if city == 'Hyderabad':
            original = hydorg['Average']
        else:
            original = tororg['Avg.Temp']
    elif mode == 'individual':
        data = data[data['year']==year]
        prompt = 'Prophet Forecast for Average Temperatures in {}, {}'
        
        if city == 'Hyderabad':
            original = hydorg[hydorg['year ']==year]['Average']
        else:
            original = tororg[tororg['year ']==year]['Avg.Temp']

    
    
    x = data['ds']
    y0 = data.yhat
    y1 = data.yhat_lower
    y2 = data.yhat_upper

    

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y0,
                        mode='lines+markers',
                        name='yhat', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=x, y=original,
                        mode='lines+markers',
                        name='original', line=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=x, y=y1,
                        name='yhat_lower', line=dict(color='green', width=2,
                              dash='dash')))
    fig.add_trace(go.Scatter(x=x, y=y2,

                        name='yhat_upper', line=dict(color='orange', width=2,
                              dash='dot')))

    fig.update_traces(marker=dict(
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

    fig.update_layout(
        title=prompt.format(city,year),
        title_x=0.5,
        xaxis_title="Months",
        yaxis_title="Average Temperature",
        legend_title="Legend",)

    return fig


@st.cache(persist=True)
def gen_seasonal_dfs(dmajcity,city):
    
    datapackage = {}
    seasondfs = []
    
    # slice by CITY
    data = dmajcity[dmajcity['City']==city]
    
    # get seasons 
    seasons = data['Seasons'].unique()
    datapackage['Seasons'] = seasons 
    
    # hemisphere 
    datapackage['Hemisphere'] = data['Hemisphere'].unique()[0]
    
    for season in seasons:
        seasondfs.append(data[data['Seasons']==season])
        
    datapackage['dfs'] = seasondfs
    
    return datapackage





def seasonality_prophet(dflist,seasons):
    models = {}
    for df in dflist:
        season = df['Seasons'].unique()[0]

        # initialise model
        print(f'{season.upper()}')
        
        model = Prophet(yearly_seasonality=True)
        
        dt_prophet = df[['dt','AverageTemperature']]
        dt_prophet.columns = ['ds','y']
        
        # fit
        model.fit(dt_prophet)
        
        # caching
        models[season] = model
        models[season+'-months'] = df['Month'].unique()
        
    return models

@st.cache(allow_output_mutation=True,suppress_st_warning=True) 
def quantify_future_seasonality(year, model_dict,seasons):
    
    quantified = {}
    
    
    season_models = model_dict
    future_dates=season_models['Winter'].make_future_dataframe(periods=2920)
    
    # my_bar = st.progress(0)


    for season in seasons:
        prediction=season_models[season].predict(future_dates)
        df = prediction
        df['year'] = df['ds'].dt.year
        df['day'] = df['ds'].dt.day
        df['month'] = df['ds'].dt.strftime("%b")

        # prune dates
        df = df[df['year']>2013]
        df = df[df['day']==1]
        df = df.drop('day',axis=1)
        df = df[df['month'].isin(season_models[f'{season}-months'])]
    
        quantified[season] = df[df['year']==year]
        # my_bar.progress(season)
    
    
    return quantified





def prophet_cs(dmajcity):

   

    mode = st.selectbox('Modes',['TimeSeries-Prophet', 'Seasonal-Prophet'],key='prophet-Modes')

    st.markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:37px; text-align:center;">{mode}<br></h1>', unsafe_allow_html=True)

    # Coverting dt to datetime obj
    dmajcity['dt'] = pd.to_datetime(dmajcity['dt'])

    if mode == 'TimeSeries-Prophet':
        cities = ['Hyderabad','Toronto']
    
        cols = st.columns(2)
        city = cols[1].selectbox('City',cities,key='casestudy')
        cols[0].markdown(f'<h1 style="font-family:{fontname0}; text-decoration:underline; font-size:30px; text-align:center;"><br>{city}<br></h1>', unsafe_allow_html=True)
            
        pred_df = prophetize(dmajcity, province=city)

        fig = project_yhat(pred_df,city=city,year=2020,mode='compound')

        st.plotly_chart(fig,use_container_width=True)

        year = st.slider('Year', min_value=2018,max_value=2020, key='yearslider')

        fig = project_yhat(pred_df,city=city,year=year,mode='individual')
        st.plotly_chart(fig,use_container_width=True)

    else:

        countries = list(dmajcity['Country'].unique())
        countries.insert(0, 'Select Country')

        cols = st.columns([3,2])
        
        country = cols[0].selectbox('Country',countries, key='select-country')

        if not country == 'Select Country':
            cities = dmajcity[dmajcity['Country']==country]['City'].unique()
        
            with cols[1].form(key='season'):

                city = st.selectbox('City', cities, key='season-sets')
        
                submit_button = st.form_submit_button(label='Forecast 🌡️')
            
            

            if submit_button:
                datapackage = gen_seasonal_dfs(dmajcity, city)
                season_models = seasonality_prophet(datapackage['dfs'], datapackage['Seasons'])
                quantified = quantify_future_seasonality(2020, model_dict=season_models, seasons=datapackage['Seasons'])



                cols = st.columns(len(quantified.keys()))


                st.markdown('***')
                
                for i, (key, value) in enumerate(zip(quantified.keys(),quantified.values()),0):

                        season = key 
                        value = value['yhat'].mean()
                        
                        cols[i].metric(label=season, value=str("%.2f" % value)+' °C',delta_color='inverse')

                # caching.clear_cache()








def cs_main():
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Temperature Dashboard",layout="wide",initial_sidebar_state="collapsed")
    st.markdown(font_url0,unsafe_allow_html=True)
    st.markdown(font_url1,unsafe_allow_html=True)

    menu_data = [
       
        {'icon': "far fa-chart-bar", 'label':"Analytics", "ttip":"Ton of Visualisations"},
        {'icon': "fa fa-cogs", 'label':"Prophet", "ttip":"Here lies the power of Machine Learning"},
        {'icon': "fa fa-blind", 'label':"Traveller?",'ttip':"The Traveller's Acquaintance"}, 
        # {'icon': "far fa-copy", 'label':"Right End"},
    ]

    # we can override any part of the primary colors of the menu
    #over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
    over_theme = {'txc_inactive': 'black','menu_background':'#F5F5F5','txc_active':'#058DFC'}
    menu_id = hc.nav_bar(menu_definition=menu_data,home_name='About',override_theme=over_theme)


    st.markdown(f"<h1 style='font-family:georgia; font-size:59px; font; text-align:center;  '>Global Temperature Variation Analysis & Modelling   <img src='data:image/png;base64,{img_to_bytes('./assets/trend.png')}' class='img-fluid' width=62 height=62><p style='text-align:left; font-weight:bold;'>| {menu_id}</p></h1>",unsafe_allow_html=True)
    st.markdown("<span id='top'></span>",unsafe_allow_html=True)
    st.markdown('***')


     # data 
    data = load_data(about=False)
    dcity, unis = load_provinces()

    # unis = load_provinces()

    


    if menu_id == 'Analytics':
        analytics_cs(data)

    elif menu_id == 'About':
        about_cs()

    elif menu_id == 'Prophet':
        prophet_cs(data)

    elif menu_id == 'Traveller?':
        traveller_cs(data, (dcity,unis))




    
   



   
if __name__ == '__main__':
    cs_main()

