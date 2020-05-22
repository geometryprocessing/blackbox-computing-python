import meshplot
import json
#import plotly.offline as plotly
#import plotly.graph_objects as go

first = True
meshplot.website()

def mp_to_md(self):
    global first
    if first:
        first = False
        res = self.to_html(imports=True, html_frame=False)
    else:
        res = self.to_html(imports=False, html_frame=False)

    return res
    
#def plotly_to_md(self):
#    print("PLOTLING")
#    html = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
#    html += "<div>"
#    html += plotly.plot(self, output_type='div', include_plotlyjs=first, show_link=False)
#    html += "</div>"

#    return html

def sp_to_md(self):
    global first
    if first:
        first = False
        res = self.to_html(imports=True, html_frame=False)
    else:
        res = self.to_html(imports=False, html_frame=False)

    return res

def lis_to_md(self):
    res = ""
    for row in self:
        for e in row:
            res += e.to_html()
    return res

get_ipython().display_formatter.formatters["text/html"].for_type(meshplot.Viewer, mp_to_md)
#get_ipython().display_formatter.formatters["text/html"].for_type(go.Figure, plotly_to_md)
#get_ipython().display_formatter.formatters["text/html"].for_type(meshplot.Subplot, sp_to_md)








