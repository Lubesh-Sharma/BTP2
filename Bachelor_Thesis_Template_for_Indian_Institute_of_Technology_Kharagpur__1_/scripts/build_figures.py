#!/usr/bin/env python3
"""Build thesis schematics as high-resolution PNG and editable draw.io documents.

The SHREC silhouettes are illustrative inputs, not correspondence results.  All
modules, text, tensors and connectors are native editable draw.io objects;
point-cloud thumbnails are embedded images.  No learned outputs are fabricated.
Run from any directory: python path/to/scripts/build_figures.py
"""
from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/btp2-thesis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

THESIS = Path(__file__).resolve().parents[1]
ROOT = THESIS.parent
OUT = THESIS / "Pictures" / "current"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "axes.unicode_minus": False,
})

C = dict(ink="#233044", muted="#637084", line="#A8B4C3", blue="#316AA2",
         blue_l="#E8F1FB", teal="#28877D", teal_l="#E8F5F1",
         purple="#8061A4", purple_l="#F0EAF8", orange="#BA7D35",
         orange_l="#FBF0DD", rose="#A3637E", rose_l="#F6EAF0", panel="#FBFCFE", border="#DCE3EB", white="#FFFFFF")


def load_points():
    path = ROOT / "input" / "SHREC" / "off" / "1.off"
    with path.open() as stream:
        assert stream.readline().strip() == "OFF"
        n = int(stream.readline().split()[0])
        pts = np.array([[float(x) for x in stream.readline().split()[:3]] for _ in range(n)])
    return pts[np.linspace(0, len(pts) - 1, min(2800, len(pts)), dtype=int)]


POINTS = load_points()


class Scene:
    def __init__(self, name, width=12, height=7.2):
        self.name, self.w, self.h = name, width, height
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
        self.ax.axis("off")
        self.mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": name, "name": name})
        model = ET.SubElement(diagram, "mxGraphModel", {"dx": str(int(width*100)), "dy": str(int(height*100)), "grid": "1", "gridSize": "10", "page": "1", "pageWidth": str(int(width*100)), "pageHeight": str(int(height*100)), "math": "0"})
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.num = 1

    def cell(self, x, y, w, h, style, value=""):
        self.num += 1
        cell = ET.SubElement(self.root, "mxCell", {"id": f"v{self.num}", "value": value,
                         "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": f"{100*x:.3f}", "y": f"{100*(self.h-y-h):.3f}",
                            "width": f"{100*w:.3f}", "height": f"{100*h:.3f}", "as": "geometry"})
        return cell

    def rect(self, x, y, w, h, fill="white", stroke=None, radius=.10, lw=1.2, z=2):
        fill, stroke = C.get(fill, fill), C.get(stroke, stroke) if stroke else "none"
        if radius:
            patch = FancyBboxPatch((x,y), w,h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=z)
        else:
            patch = Rectangle((x,y),w,h,facecolor=fill,edgecolor=stroke,linewidth=lw,zorder=z)
        self.ax.add_patch(patch)
        self.cell(x,y,w,h,f"rounded={int(bool(radius))};arcSize=12;fillColor={fill};strokeColor={stroke};strokeWidth={lw};")

    def text(self, x, y, s, size=15, color="ink", weight="normal", ha="center", va="center", w=None, h=None, z=8):
        color="#000000"
        rendered=re.sub(r"\b([LFZH])_([A-Za-z]+)\b", lambda m: "$"+m[1]+r"_{\mathrm{"+m[2]+"}}$", s)
        rendered=rendered.replace("Eθ", r"$E_\theta$")
        self.ax.text(x,y,rendered,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.25,zorder=z)
        lines=s.split("\n")
        w=w or max(.25,max(len(a) for a in lines)*size/105)
        h=h or len(lines)*size/58
        x0=x if ha=="left" else x-w if ha=="right" else x-w/2
        y0=y-h if va=="top" else y if va=="bottom" else y-h/2
        # px font scaling matches 100px / inch geometry at 72pt / inch.
        self.cell(x0,y0,w,h,f"text;html=0;align={ha};verticalAlign=middle;whiteSpace=wrap;overflow=visible;strokeColor=none;fillColor=none;fontFamily=Helvetica;fontSize={size*100/72:.2f};fontColor={color};fontStyle={int(weight=='bold')};",s)

    def line(self, points, color="line", lw=1.5, dash=False, arrow=False, z=3):
        color="#555555"
        p=np.asarray(points)
        self.ax.plot(p[:,0],p[:,1],color=color,lw=lw,linestyle=(0,(4,3)) if dash else "-",zorder=z)
        if arrow:
            self.ax.add_patch(FancyArrowPatch(p[-2],p[-1],arrowstyle="-|>",mutation_scale=13,linewidth=lw,color=color,zorder=z))
        self.num+=1
        edge=ET.SubElement(self.root,"mxCell",{"id":f"e{self.num}","value":"","style":f"edgeStyle=none;rounded=0;html=0;endArrow={'block' if arrow else 'none'};endFill=1;strokeColor={color};strokeWidth={lw};dashed={int(dash)};","edge":"1","parent":"1"})
        geo=ET.SubElement(edge,"mxGeometry",{"relative":"1","as":"geometry"})
        for q, role in [(p[0],"sourcePoint"),(p[-1],"targetPoint")]:
            ET.SubElement(geo,"mxPoint",{"x":str(100*q[0]),"y":str(100*(self.h-q[1])),"as":role})
        if len(p)>2:
            arr=ET.SubElement(geo,"Array",{"as":"points"})
            for q in p[1:-1]: ET.SubElement(arr,"mxPoint",{"x":str(100*q[0]),"y":str(100*(self.h-q[1]))})

    def circle(self,x,y,r,color="blue",fill=None,lw=1.2):
        color=C.get(color,color); fill=C.get(fill,fill) if fill else color
        self.ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=color,lw=lw,zorder=4))
        self.cell(x-r,y-r,2*r,2*r,f"ellipse;fillColor={fill};strokeColor={color};strokeWidth={lw};")

    def panel(self,x,y,w,h,label,title,color="blue"):
        self.rect(x,y,w,h,"panel","border",radius=.12,lw=.9,z=0)
        self.text(x+.18,y+h-.29,label,size=17,color=color,weight="bold",ha="left")
        self.text(x+.63,y+h-.29,title,size=17,weight="bold",ha="left")

    def module(self,x,y,w,h,title,subtitle=None,color=None,size=17):
        if color is None:
            t=title.lower()
            if any(v in t for v in ('dino','visual','project')): color='orange'
            elif any(v in t for v in ('orientation','yaw','angular','bin ','domain','mask','synthetic')): color='purple'
            elif any(v in t for v in ('decoder','fusion','hadm','graph','spectral','heat','normalization','preprocessing')): color='teal'
            else: color='blue'
        self.rect(x,y,w,h,color+'_l',color,radius=.08,lw=1.05)
        if subtitle:
            self.text(x+w/2,y+h*.66,title,size=size,color=color,weight="bold")
            self.text(x+w/2,y+h*.28,subtitle,size=max(12,size-3),color="muted")
        else: self.text(x+w/2,y+h/2,title,size=size,color=color,weight="bold")

    def tensor(self,x,y,w=.55,h=.75,color="blue",rows=7,cols=4):
        # Offset tensor slices communicate channels; every cell stays editable.
        self.rect(x+.10,y+.10,w,h,color+"_l",color,radius=0,lw=.65)
        self.rect(x+.05,y+.05,w,h,color+"_l",color,radius=0,lw=.65)
        self.rect(x,y,w,h,"white",color,radius=0,lw=.7)
        rng=np.random.default_rng(7)
        for i in range(rows):
            for j in range(cols):
                a=.20+.62*rng.random()
                rgb=matplotlib.colors.to_rgb(C[color]); rgb=tuple(1-a+a*v for v in rgb)
                self.rect(x+j*w/cols,y+i*h/rows,w/cols,h/rows,matplotlib.colors.to_hex(rgb),None,radius=0)

    def cloud(self,x,y,w,h,color="blue",angle=0,alpha=.72,nodes=0):
        p=POINTS.copy(); a=np.deg2rad(angle)
        xx=p[:,0]*np.cos(a)+p[:,2]*np.sin(a)
        xy=np.column_stack((xx,p[:,1]))
        xy-=xy.min(0); scale=min(w/max(xy[:,0].max(),1e-6),h/max(xy[:,1].max(),1e-6)); xy*=scale
        xy[:,0]+=(w-xy[:,0].max())/2; xy[:,1]+=(h-xy[:,1].max())/2
        self.ax.scatter(x+xy[:,0],y+xy[:,1],s=1.0,c=C[color],alpha=alpha,linewidths=0,zorder=4,rasterized=False)
        fig=plt.figure(figsize=(w,h),dpi=180); ax=fig.add_axes((0,0,1,1)); ax.axis("off"); ax.set(xlim=(0,w),ylim=(0,h))
        ax.scatter(xy[:,0],xy[:,1],s=1,c=C[color],alpha=alpha,linewidths=0)
        buf=io.BytesIO(); fig.savefig(buf,format="png",transparent=True,dpi=180); plt.close(fig)
        image="data:image/png,"+base64.b64encode(buf.getvalue()).decode("ascii")
        self.cell(x,y,w,h,f"shape=image;aspect=fixed;imageAspect=0;image={image};")
        if nodes:
            idx=np.linspace(0,len(xy)-1,nodes,dtype=int)
            q=xy[idx]
            for i in range(len(q)):
                distances=((q-q[i])**2).sum(1)
                for j in np.argsort(distances)[1:3]:
                    if i<j:self.line([(x+q[i,0],y+q[i,1]),(x+q[j,0],y+q[j,1])],color="orange",lw=.8,z=5)
            for u,v in q:self.circle(x+u,y+v,.027,color="orange",fill="white",lw=.9)

    def footer(self,text):
        self.text(.18,.22,text,size=12,color="muted",ha="left")

    def save(self):
        self.fig.savefig(OUT/f"{self.name}.png",dpi=300)
        ET.indent(self.mxfile)
        ET.ElementTree(self.mxfile).write(OUT/f"{self.name}.drawio",encoding="utf-8",xml_declaration=True)
        plt.close(self.fig)


def axes(s,x,y,scale=.25):
    for dx,dy,label in [(scale,0,'x'),(0,scale,'y'),(-scale*.6,-scale*.5,'z')]:
        s.line([(x,y),(x+dx,y+dy)],lw=.8,arrow=True)
        s.text(x+dx*1.25,y+dy*1.25,label,size=10)


def attr_cloud(s,x,y,w,h,angle=0,mesh=False):
    """Illustrate attribute transfer with known identity on a rotated input."""
    from matplotlib.colors import LinearSegmentedColormap
    pts=POINTS.copy(); a=np.deg2rad(angle)
    xy=np.column_stack((pts[:,0]*np.cos(a)+pts[:,2]*np.sin(a),pts[:,1]))
    v=(pts[:,1]-pts[:,1].min())/(np.ptp(pts[:,1])+1e-12)
    cmap=LinearSegmentedColormap.from_list('attribute',['#426A87','#D6D6CE','#AE794A'])
    colors=cmap(v)
    xy-=xy.min(0); xy*=min(w/xy[:,0].max(),h/xy[:,1].max())
    xy[:,0]+=(w-xy[:,0].max())/2; xy[:,1]+=(h-xy[:,1].max())/2
    s.ax.scatter(x+xy[:,0],y+xy[:,1],s=1,c=colors,linewidths=0,zorder=4)
    fig=plt.figure(figsize=(w,h),dpi=180); ax=fig.add_axes((0,0,1,1));ax.axis('off');ax.set(xlim=(0,w),ylim=(0,h))
    ax.scatter(xy[:,0],xy[:,1],s=1,c=colors,linewidths=0)
    buf=io.BytesIO();fig.savefig(buf,format='png',transparent=True);plt.close(fig)
    s.cell(x,y,w,h,'shape=image;imageAspect=0;image=data:image/png,'+base64.b64encode(buf.getvalue()).decode()+';')


def matrix(s,x,y,w=1.5,h=1.1,masked=False,label=None):
    rows,cols=8,8
    rng=np.random.default_rng(5)
    for i in range(rows):
        for j in range(cols):
            level=.2+.65*rng.random()
            fill=matplotlib.colors.to_hex((1-level*.55,1-level*.35,1-level*.22)) if j<5 else '#DADADA'
            if masked and j<5 and (i+j)%3==0:fill='#FFFFFF'
            s.rect(x+j*w/cols,y+i*h/rows,w/cols,h/rows,fill,'#FFFFFF',radius=0,lw=.25)
    s.rect(x,y,w,h,'none','#777777',radius=0,lw=.7)
    s.line([(x+w*5/8,y),(x+w*5/8,y+h)],lw=.9)
    s.text(x+w*5/16,y+h+.18,'H',size=12)
    for j,name in enumerate(('x','y','z')):s.text(x+w*(5.5+j)/8,y+h+.18,name,size=11)
    if label:s.text(x+w/2,y-.22,label,size=13)


def loss(s,x,y,w,h,text,size=12):
    s.rect(x,y,w,h,'orange_l','orange',radius=.06,lw=.9)
    s.text(x+w/2,y+h/2,text,size=size)


def mesh_cloud(s,x,y,w,h):
    # Mesh thumbnail uses actual input triangles, projected into the image plane.
    from matplotlib.collections import LineCollection
    path=ROOT/'input/SHREC/off/1.off'
    with path.open() as f:
        f.readline(); n,nf,*_=map(int,f.readline().split())
        pts=np.array([[float(v) for v in f.readline().split()[:3]] for _ in range(n)])
        faces=[list(map(int,f.readline().split()))[1:] for _ in range(nf)]
    xy=pts[:,:2].copy();xy-=xy.min(0);xy*=min(w/xy[:,0].max(),h/xy[:,1].max());xy[:,0]+=(w-xy[:,0].max())/2
    segments=[]
    for face in faces[::2]:
        for i in range(len(face)):segments.append(xy[[face[i],face[(i+1)%len(face)]]])
    seg=np.array(segments)
    s.ax.add_collection(LineCollection(seg+np.array([x,y]),colors='#688397',linewidths=.15,alpha=.55))
    fig=plt.figure(figsize=(w,h),dpi=180); ax=fig.add_axes((0,0,1,1));ax.set(xlim=(0,w),ylim=(0,h));ax.axis('off')
    ax.add_collection(LineCollection(seg,colors='#688397',linewidths=.15,alpha=.55))
    buf=io.BytesIO();fig.savefig(buf,format='png',transparent=True);plt.close(fig)
    s.cell(x,y,w,h,'shape=image;imageAspect=0;image=data:image/png,'+base64.b64encode(buf.getvalue()).decode()+';')


def overview():
    s=Scene('pipeline-overview',12,8.0)
    # Full-sized augmented input and encoder; compact numbered loss boxes.
    s.cloud(2.2,6.75,.65,1.0,'blue',angle=15)
    s.text(2.52,7.88,'Augmented subset X′',size=11)
    s.module(3.3,6.95,1.9,.65,'Same ASMAE Eθ',size=13)
    s.line([(2.95,7.28),(3.3,7.28)],arrow=True)
    loss(s,6.0,7.05,.8,.45,'L₁',size=13)
    s.line([(5.2,7.28),(6.0,7.28)],dash=True,arrow=True,lw=1.0)
    loss(s,6.0,6.2,.8,.45,'L₂',size=13)

    s.cloud(.12,3.5,.72,1.4,'blue');s.cloud(1.01,3.5,.72,1.4,'blue',angle=50)
    s.text(.48,5.14,'Source X',size=13);s.text(1.37,5.14,'Target Y',size=13)
    axes(s,.34,3.25,.22)
    s.module(2.3,4.4,1.9,.95,'Preprocessing','H + x, y, z',size=14)
    s.module(2.3,3.25,1.9,.8,'Orientation',size=14)
    s.line([(1.8,4.85),(2.3,4.85)],arrow=True)
    s.line([(3.25,4.4),(3.25,4.05)],arrow=True)
    s.text(3.62,4.22,'XYZ',size=10)
    s.module(5.0,3.15,1.9,1.0,'ASMAE Eθ','Point features',size=15)
    s.line([(4.2,3.65),(5.0,3.65)],arrow=True)
    s.text(4.6,3.45,'align',size=10)
    s.line([(4.2,4.65),(4.4,4.65),(4.4,3.9),(5.0,3.9)],arrow=True)
    s.text(4.55,4.4,'H',size=11)
    s.module(7.95,3.15,1.85,1.0,'Fusion + HADM','Dense matching',size=14)
    s.line([(6.9,3.65),(7.95,3.65)],arrow=True)
    attr_cloud(s,10.1,3.13,.68,1.40)
    attr_cloud(s,11.1,3.13,.68,1.40,angle=25)
    s.line([(9.8,3.65),(10.05,3.65)],arrow=True)
    s.line([(10.82,3.8),(11.07,3.8)],arrow=True)
    s.text(10.96,4.88,'Point-to-point map',size=13)
    s.text(10.96,2.85,'Colour transfer',size=13)
    s.module(5.0,4.8,1.9,.7,'Cross-decoder',size=14)
    s.line([(5.95,4.15),(5.95,4.8)],arrow=True)
    loss(s,7.95,4.8,1.85,.7,'L_rec')
    s.line([(6.9,5.15),(7.95,5.15)],arrow=True,dash=True)
    loss(s,1.3,2.2,.75,.45,'L₃',size=13)
    s.line([(2.6,3.25),(2.12,3.25),(2.12,2.425),(2.05,2.425)],dash=True,arrow=True,lw=1.0)
    for d in (.14,.07,0):s.rect(.4+d,.53+d,1.02,1.05,'#F7F7F7','#777777',radius=.02,lw=.7)
    s.cloud(.69,.6,.45,.9,'blue')
    s.text(.97,.26,'Projected views',size=12)
    s.module(2.3,.8,1.9,.85,'Frozen DINOv2',color='rose',size=14)
    s.line([(1.65,1.18),(2.3,1.18)],arrow=True)
    s.module(5.0,.8,1.9,.85,'DINO head',color='rose',size=14)
    s.line([(5.95,3.15),(5.95,1.65)],arrow=True)
    s.text(6.45,2.25,'predict',size=11)
    loss(s,7.95,.8,1.85,.85,'L_DINO')
    s.line([(6.9,1.22),(7.95,1.22)],arrow=True,dash=True)
    s.line([(4.2,1.0),(4.5,1.0),(4.5,.4),(8.88,.4),(8.88,.8)],dash=True,arrow=True)
    s.line([(4.2,1.35),(4.55,1.35),(4.55,2.65),(8.5,2.65),(8.5,3.15)],arrow=True)
    s.text(7.4,2.85,'observed visual features',size=10)
    s.line([(6.9,1.5),(7.3,1.5),(7.3,2.32),(9.3,2.32),(9.3,3.15)],arrow=True)
    s.text(8.7,2.08,'predicted visual features',size=10)

    # Intrinsic descriptors and clean features supervise the compact losses.
    s.line([(4.0,5.35),(4.0,6.425),(6.0,6.425)],dash=True,arrow=True,lw=1.0)
    s.text(4.17,5.7,'ψ',size=10)
    s.line([(5.0,4.0),(4.72,4.0),(4.72,5.85),(7.15,5.85),(7.15,7.28),(6.8,7.28)],dash=True,arrow=True,lw=1.0)
    s.line([(7.15,6.425),(6.8,6.425)],dash=True,arrow=True,lw=1.0)
    s.text(6.15,5.67,'Clean embeddings',size=9)
    s.text(7.35,7.0,'Reuse clean Z_X',size=10,ha='left')
    s.save()


def preprocessing():
    s=Scene('preprocessing',12,5.9)
    mesh_cloud(s,.32,3.72,.85,1.4);s.text(.75,5.5,'Surface',size=14)
    s.cloud(2.1,3.72,.85,1.4,'blue');s.text(2.52,5.5,'Point cloud',size=14)
    axes(s,2.1,3.55,.23)
    s.line([(1.3,4.4),(1.95,4.4)],arrow=True)
    s.cloud(4.0,3.72,.85,1.4,'blue',nodes=25);s.text(4.42,5.5,'Neighbour graph',size=14)
    s.line([(3.08,4.4),(3.85,4.4)],arrow=True)
    s.module(5.65,4.0,2.1,.85,'Graph Laplacian','L = D − A',size=14)
    s.line([(4.98,4.4),(5.65,4.4)],arrow=True)
    s.module(8.48,4.0,2.6,.85,'Spectral basis','Eigenpairs (λ, φ)',size=14)
    s.line([(7.75,4.4),(8.48,4.4)],arrow=True)
    s.module(5.65,1.55,2.1,.85,'Heat diffusion','Landmark responses',size=14)
    s.line([(9.78,4.0),(9.78,3.25),(6.7,3.25),(6.7,2.4)],arrow=True)
    s.text(8.4,3.53,'Graph spectrum',size=12)
    matrix(s,8.65,1.12,2.7,1.48,label='F = [ H | x | y | z ]')
    s.line([(7.75,1.98),(8.65,1.98)],arrow=True)
    s.text(9.5,2.98,'HKS-based matrix',size=13)
    s.module(1.4,1.55,2.8,.85,'Normalization','x, y, z',size=14)
    s.line([(2.55,3.45),(2.55,2.4)],arrow=True)
    s.line([(4.2,1.85),(4.6,1.85),(4.6,.52),(10.85,.52),(10.85,1.12)],arrow=True)
    s.text(6.7,.82,'Coordinate channels',size=12)
    s.save()


def orientation():
    s=Scene('orientation-module',12,6.2)
    s.cloud(.15,3.8,.7,1.45,'blue');s.cloud(1.0,3.8,.7,1.45,'blue',angle=55)
    s.text(.9,5.65,'Subsets of X and Y',size=13)
    axes(s,.34,3.5,.22)
    s.module(2.1,4.0,1.65,1.05,'Point encoder','Point features',size=14)
    s.line([(1.78,4.52),(2.1,4.52)],arrow=True)
    s.module(4.25,4.0,1.7,1.05,'Interaction','Pair features',size=14)
    s.line([(3.75,4.52),(4.25,4.52)],arrow=True)
    s.module(6.55,4.75,1.9,.7,'Bin confidence',size=14)
    s.module(6.55,3.25,1.9,.85,'Angular residual','cos δ, sin δ',size=14)
    s.line([(5.95,4.52),(6.2,4.52),(6.2,5.1),(6.55,5.1)],arrow=True)
    s.line([(6.2,4.52),(6.2,3.68),(6.55,3.68)],arrow=True)
    s.module(9.0,4.0,1.4,1.05,'Refined yaw','θ = c + δ',size=14)
    s.line([(8.45,5.1),(8.7,5.1),(8.7,4.7),(9.0,4.7)],arrow=True)
    s.line([(8.45,3.68),(8.7,3.68),(8.7,4.3),(9.0,4.3)],arrow=True)
    s.cloud(10.95,3.8,.75,1.45,'blue')
    s.line([(10.4,4.52),(10.9,4.52)],arrow=True)
    s.text(11.1,5.65,'Aligned Y',size=13)
    s.text(10.95,3.4,'Full cloud',size=12)
    s.module(.25,1.4,2.15,.85,'Synthetic rotations','Known relative angles',size=14)
    loss(s,6.55,1.4,2.15,.85,'L_bin + L_res')
    s.line([(2.4,1.85),(3.0,1.85),(3.0,2.7),(7.62,2.7),(7.62,2.25)],arrow=True,dash=True)
    s.line([(8.45,4.95),(8.85,4.95),(8.85,1.85),(8.7,1.85)],arrow=True,dash=True)
    s.module(3.65,.3,2.15,.85,'Domain classifier','Gradient reversal',size=13)
    s.line([(5.1,4.0),(5.1,3.05),(5.72,3.05),(5.72,1.15)],arrow=True)
    loss(s,6.55,.3,2.15,.65,'L_domain')
    s.line([(5.8,.73),(6.55,.73)],arrow=True,dash=True)
    loss(s,9.45,1.4,2.15,.85,'L_subset')
    s.line([(8.45,3.5),(8.85,3.5)],dash=True)
    s.line([(8.85,1.85),(9.45,1.85)],arrow=True,dash=True)
    s.text(10.53,1.02,'Repeated subsets',size=12)
    s.save()


def asmae():
    s=Scene('asmae-pipeline',12,6.4)
    # A: the two mask-selection branches from the original AAMG diagram.
    matrix(s,.4,4.3,1.5,1.05,label='Source F_X')
    s.module(2.55,5.1,2.0,.65,'Light DGCNN','',color='purple',size=14)
    s.module(5.05,5.1,2.0,.65,'Query scores',color='purple',size=14)
    s.module(7.55,5.1,2.05,.65,'Gumbel selection',color='purple',size=13)
    s.text(6.05,5.95,'Query vectors',size=12)
    s.line([(6.05,5.82),(6.05,5.75)],arrow=True)
    s.line([(1.9,4.85),(2.2,4.85),(2.2,5.42),(2.55,5.42)],arrow=True)
    s.line([(4.55,5.42),(5.05,5.42)],arrow=True)
    s.line([(7.05,5.42),(7.55,5.42)],arrow=True)
    s.module(2.55,3.85,2.0,.85,'Column similarity','H channels only',color='purple',size=13)
    s.module(5.05,3.85,2.0,.85,'Antipodal seeds','Random + least similar',color='purple',size=13)
    s.module(7.55,3.85,2.05,.85,'Channel clusters','Union around A and B',color='purple',size=13)
    s.line([(2.2,4.85),(2.2,4.27),(2.55,4.27)],arrow=True)
    s.line([(4.55,4.27),(5.05,4.27)],arrow=True)
    s.line([(7.05,4.27),(7.55,4.27)],arrow=True)
    s.module(10.15,4.35,1.3,.9,'AND','M = p ∧ c',color='purple',size=14)
    s.line([(9.6,5.42),(10.8,5.42),(10.8,5.25)],arrow=True)
    s.text(10.7,5.73,'Point mask p',size=11)
    s.line([(9.6,4.27),(9.88,4.27),(9.88,4.6),(10.15,4.6)],arrow=True)
    s.text(10.98,3.87,'Channel\nmask c',size=10,ha='left')
    s.text(5.8,3.48,'Zero entries selected by M; retain x, y, z',size=12)
    s.line([(10.8,4.35),(10.8,3.18),(.24,3.18),(.24,2.3),(.4,2.3)],arrow=True)
    s.text(11.15,3.48,'M',size=12)
    # B: masked reconstruction, paired clean context, same encoder parameters.
    matrix(s,.4,1.9,1.5,.8,masked=True,label='Masked source')
    matrix(s,.4,0.45,1.5,.8,label='Clean target')
    s.module(2.6,1.9,1.9,.8,'Encoder Eθ',color='blue',size=15)
    s.module(2.6,0.45,1.9,.8,'Encoder Eθ',color='blue',size=15)
    s.line([(1.9,2.3),(2.6,2.3)],arrow=True)
    s.line([(1.9,0.85),(2.6,0.85)],arrow=True)
    s.line([(3.55,1.25),(3.55,1.9)],dash=True)
    s.text(4.3,1.58,'same θ',size=11)
    s.tensor(5.1,2.03,.42,.5,'blue',rows=5,cols=3)
    s.tensor(5.1,0.58,.42,.5,'teal',rows=5,cols=3)
    s.line([(4.5,2.3),(5.1,2.3)],arrow=True)
    s.line([(4.5,0.85),(5.1,0.85)],arrow=True)
    s.module(6.4,1.12,1.9,1.0,'Cross-attention','Decoder + head',color='teal',size=14)
    s.line([(5.62,2.3),(6.0,2.3),(6.0,1.85),(6.4,1.85)],arrow=True)
    s.line([(5.62,0.85),(6.0,0.85),(6.0,1.4),(6.4,1.4)],arrow=True)
    s.text(5.95,2.72,'Q',size=13);s.text(6.0,0.43,'K, V',size=13)
    s.tensor(8.85,1.25,.8,.65,'teal',rows=6,cols=4)
    s.line([(8.3,1.62),(8.85,1.62)],arrow=True)
    s.text(9.25,2.35,'Reconstruct H_X',size=12)
    loss(s,10.25,1.25,1.15,.7,'L_rec')
    s.line([(9.75,1.62),(10.25,1.62)],arrow=True,dash=True)
    s.text(9.9,0.7,'Compare original masked entries',size=11)
    s.save()


if __name__=='__main__':
    overview(); preprocessing(); orientation(); asmae()
    names={'pipeline-overview','preprocessing','orientation-module','asmae-pipeline'}
    for f in OUT.iterdir():
        if f.is_file() and (f.stem not in names or f.suffix not in {'.png','.drawio'}):f.unlink()
    print('Built four PNG figures and editable draw.io sources.')
