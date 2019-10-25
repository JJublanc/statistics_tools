def plot_cumul(x2,y2,name = 'F(t)'):
    return  go.Scatter(
        x=x2,
        y=y2,
        fill='tozeroy',
        mode= 'none',
        name = name)

def plot_density(x1,y1,name = 'f(t)'):
    return go.Scatter(
            x=x1,
            y=y1,
            name = name)

def plot_text(txt=["text"], x_txt=[0], y_txt=[0.08],size=35,color="#291E94"):
    return go.Scatter(
        x=x_txt,
        y=y_txt,
        mode='text',
        text=txt,
        textposition='bottom center',
        showlegend=False,
            textfont=dict(
            family='sans serif',
            size=size,
            color=color))
        
def layout_annoted(t):
        return go.Layout(width=900,height=500,
                           annotations=[
                            dict(
                                x=t,
                                y=0,
                                xref='x',
                                yref='y',
                                text='t = {}'.format(round(t,3)),
                                showarrow=True,
                                font=dict(
                                    family='Courier New, monospace',
                                    size=16,
                                    color='#ffffff'
                                ),
                                align='center',
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor='#636363',
                                ax=20,
                                ay=-30,
                                bordercolor='#c7c7c7',
                                borderwidth=2,
                                borderpad=4,
                                bgcolor='#ff7f0e',
                                opacity=0.8
                                )
                            ]
                        )


def get_cdf_plot_values(loi,t):
    cdf = loi.cdf(t)
    text = "risque 1 = {}%".format(round(100 - round(cdf*100,1),1))
    text_alter = "risque 2 = {}%".format(round(cdf*100,1),1)
    min_ = loi.ppf(0.001)
    max_ = loi.ppf(0.999)
    
    x1=np.linspace(min_, max_, int((max_ - min_)*500))
    y1=loi.pdf(x1)

    x2 = x1[x1<t]
    y2 = y1[x1<t]
    
    x3 = x1[x1>t]
    y3 = y1[x1>t]
    
    x_txt = (max_ + min_)/2
    return cdf, text, min_, max_, x1, x2, x3, y1, y2, y3, x_txt, text_alter


def plot_cdf(t,loi, threshold = False):
    
    cdf, text, min_, max_, x1, x2, x3, y1, y2, y3, x_txt, text_alter = get_cdf_plot_values(loi,t)
    
    trace1 = plot_density(x1,y1)
    trace2 = plot_cumul(x3,y3)
    trace3 = plot_text(txt=[text], x_txt=[t], y_txt=[max(y1)*1.2],size=35)
    
    layout = layout_annoted(t)
    
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    
    
    if threshold:
        fig["layout"]["shapes"] = [{'type': 'line',
                              'x0': threshold,
                              'y0': 0,
                              'x1': threshold,
                              'y1':max(y1),
                              'line': {'color': 'rgb(222, 0, 0)',
                                       'width': 3,},}]
    iplot(fig, filename='')
    
    return fig

def plot_cdf_2(t,loi_1,loi_2, threshold=False, annoted=False):
    cdf1, text1, min_1, max_1, x11, x12, x13, y11, y12, y13, x_txt1, text_alter1 = get_cdf_plot_values(loi1,t)
    cdf2, text2, min_2, max_2, x21, x22, x23, y21, y22, y23, x_txt2, text_alter2 = get_cdf_plot_values(loi2,t)

    trace1 = plot_density(x11,y11,name="H_0")
    trace2 = plot_cumul(x13,y13,"risque 1")
    trace3 = plot_text(txt=text1, x_txt=[t + (max_2 - min_1)/8], y_txt=[y13[0]/5 + 0.1],size=25, color = "#FB7900")
    
    trace4 = plot_density(x21,y21,"H_1")
    trace5 = plot_cumul(x22,y12,"risque2")
    trace6 = plot_text(txt=[text_alter2], x_txt=[t - (max_2 - min_1)/8], y_txt=[y23[0]/5 + 0.1],size=25, color = "purple")

    
    if annoted :
        layout = layout_annoted(t)
    else : 
        layout = go.Layout(showlegend=False, width=900,height=500)
    
    if threshold:
        fig["layout"]["shapes"] = [{'type': 'line',
                              'x0': threshold,
                              'y0': 0,
                              'x1': threshold,
                              'y1':max(y11),
                              'line': {'color': 'rgb(222, 0, 0)',
                                       'width': 3,},}]
    
    fig = go.Figure(data=[trace1, trace2, trace3, trace4, trace5, trace6], layout=layout)
    iplot(fig, filename='')
    
def plot_cdf_2_lois(loi1, loi2, threshold=False, annoted=False):
    def f(t):
        plot_cdf_2(t,loi1, loi2, threshold, annoted=False)
    return f

def plot_ppf(p,loi, threshold=False):
    t = loi.ppf(p)
    return plot_cdf(t,loi,threshold)

    
def plot_cdf_loi(loi,threshold=False):
    def f(t):
        plot_cdf(t,loi,threshold)
    return f


def plot_ppf_loi(loi):
    def f(p):
        plot_ppf(p,loi)
    return f

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_proba(S, 
               label_pos, 
               limite_name, 
               stat_name="S", 
               sym = False,
               message = True,
               message_1 = False,
               message_2 = False,
               fontsize_message_1 = 10,
               fontsize_message_2 = 10,
               loi = scs.norm(0,1)):
    
    sigma = loi.std()
    mean = loi.mean()
    a = S
    b = a + sigma*8
    
    if not message :
        message_1 = ""
        message_2 = ""
    else:
        if not message_1:
            message_1 = r"$P({} > ${})".format(stat_name,limite_name)
        if not message_2:
            message_2 = r"$P({} < -${})".format(stat_name,limite_name)

    
    # integral limits
    x = np.linspace(mean-4*sigma,mean+4*sigma)
    y = loi.pdf(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'r', linewidth=2)
    ax.set_ylim(bottom=0)

    # Make the shaded region
    ix = np.linspace(a, b)
    iy = loi.pdf(ix)
    verts = [(a, 0), *zip(ix, iy), (b, 0)]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.5')
    ax.add_patch(poly)
    
    ax.text(label_pos, 0.15 * (np.max(y)), message_1,
            horizontalalignment='center', fontsize=fontsize_message_1)


    if sym :
        ix_sym = np.linspace(mean-b, mean-a)
        iy_sym = loi.pdf(ix_sym)
        verts_sym = [(mean-b, 0), *zip(ix_sym, iy_sym), (mean-a, 0)]
        poly_sym = Polygon(verts_sym, facecolor='0.8', edgecolor='0.5')
        ax.add_patch(poly_sym)
        
        ax.text(mean-label_pos, 0.15 * (np.max(y)), message_2 ,
            horizontalalignment='center', fontsize=fontsize_message_2)

    fig.text(0.9, 0.05, 'Values')
    fig.text(0.1, 0.9, 'Density')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xticks([])
    ax.set_xticklabels(('$a$', '$b$'))
    ax.set_yticks([])