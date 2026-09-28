import io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

st.set_page_config(page_title='Integrated Freeze-Drying Digital Twin', page_icon='❄️', layout='wide')
R_GAS=8.314462618; TORR_PA=133.322368; CM2_M2=1e-4; G_KG=1e-3; HR_S=3600.

# ========================= SHARED PHYSICS =========================
def pice(Tc): return np.exp(-6144.96/(Tc+273.15)+24.01849)
def pwater(Tk):
    Tc=Tk-273.15
    return 611.21*np.exp((18.678-Tc/234.5)*Tc/(257.14+Tc))
def rp(Lcm,p): return p['R0']+p['A1']*Lcm/(1+p['A2']*Lcm)
def recipe_value(recipe,t):
    q=recipe[(recipe['Start (h)']<=t)&(t<recipe['End (h)'])]
    if q.empty:q=recipe.iloc[[-1]]
    q=q.iloc[0]; return float(q['Shelf T (°C)']),float(q['Pressure (Torr)'])
def tri(a,b,c,d):
    n=len(d);cp=np.zeros(n-1);dp=np.zeros(n);cp[0]=c[0]/b[0];dp[0]=d[0]/b[0]
    for i in range(1,n-1):
        den=b[i]-a[i-1]*cp[i-1];cp[i]=c[i]/den;dp[i]=(d[i]-a[i-1]*dp[i-1])/den
    dp[-1]=(d[-1]-a[-1]*dp[-2])/(b[-1]-a[-1]*cp[-1]);x=np.zeros(n);x[-1]=dp[-1]
    for i in range(n-2,-1,-1):x[i]=dp[i]-cp[i]*x[i+1]
    return x
def ceq(aw,T,p):
    af=p['Af0']*np.exp(p['Ef1']/T);al=p['AL0']*np.exp(p['Ef2']/T);bl=p['BL0']*np.exp(p['Ef3']/T)
    return np.clip(af*aw**p['n_iso']+al*aw/(1+bl*aw),0,1)
def kdes(T,p): return p['Ades']*np.exp(-p['Edes']/(R_GAS*T))
def Tg(C,p):
    w=np.clip(C/(1+C),0,.5);d=1-w
    base=(d*p['Tgd']+p['Kgt']*w*p['Tgw'])/(d+p['Kgt']*w)
    return base+p['g1']*w*d+p['g2']*w*w*d

def default_recipe(kind):
    if kind=='Phase 3':
        return pd.DataFrame({'Start (h)':[0.,10.,40.,55.],'End (h)':[10.,40.,55.,120.],'Shelf T (°C)':[-20.,-8.,10.,30.],'Pressure (Torr)':[.10,.10,.08,.05]})
    return pd.DataFrame({'Start (h)':[0.,4.,5.,7.,12.],'End (h)':[4.,5.,7.,12.,100.],'Shelf T (°C)':[-8.,-3.,-3.,-11.,-8.],'Pressure (Torr)':[.10,.10,.15,.05,.10]})

def shared_inputs(prefix):
    st.sidebar.subheader('Geometry and formulation')
    fill=st.sidebar.number_input('Fill volume (mL)',48.,key=prefix+'fill')
    dout=st.sidebar.number_input('Outer diameter (cm)',4.7,key=prefix+'dout')
    wall=st.sidebar.number_input('Wall thickness (cm)',.17,key=prefix+'wall')
    solid=st.sidebar.number_input('Solid fraction',.0625,key=prefix+'solid')
    rhoice=st.sidebar.number_input('Ice density (kg/m³)',918.,key=prefix+'rhoice')
    dH=st.sidebar.number_input('Sublimation heat (kJ/kg)',2834.6,key=prefix+'dH')*1000
    Tcrit=st.sidebar.number_input('Critical product T (°C)',-10.,key=prefix+'Tcrit')
    st.sidebar.subheader('Product resistance')
    R0=st.sidebar.number_input('R0 (cm²·h·Torr/g)',44.59,key=prefix+'R0')
    A1=st.sidebar.number_input('A1',1451.73,key=prefix+'A1');A2=st.sidebar.number_input('A2',12.93,key=prefix+'A2')
    inner=dout-2*wall
    if inner<=0:st.error('Outer diameter must exceed twice the wall thickness.');st.stop()
    area_cm2=math.pi*inner**2/4;area=area_cm2*CM2_M2;H=(fill/area_cm2)/100
    return dict(fill=fill,dout=dout,wall=wall,inner=inner,solid=solid,rhoice=rhoice,dH=dH,Tcrit=Tcrit,R0=R0,A1=A1,A2=A2,area_cm2=area_cm2,area=area,H=H)

def property_inputs(prefix):
    with st.sidebar.expander('Paper material and boundary defaults'):
        rhof=st.number_input('Frozen density (kg/m³)',927.1,key=prefix+'rhof');cpf=st.number_input('Frozen Cp (J/kg/K)',1940.,key=prefix+'cpf');kf=st.number_input('Frozen k (W/m/K)',2.763,key=prefix+'kf')
        rhod=st.number_input('Dry density (kg/m³)',55.,key=prefix+'rhod');cpd=st.number_input('Dry Cp (J/kg/K)',259.5,key=prefix+'cpd');kd0=st.number_input('Dry k intercept',.02706,key=prefix+'kd0');kdP=st.number_input('Dry k/Pa',8.826e-5,format='%.8f',key=prefix+'kdP')
        htop=st.number_input('Top h (W/m²/K)',4.2,key=prefix+'htop');hgap=st.number_input('Bottom gap h',8.73,key=prefix+'hgap');hcontact=st.number_input('Bottom contact h',16.22,key=prefix+'hcontact');contact=st.number_input('Contact fraction',.278,key=prefix+'contact');hside=st.number_input('Edge side h',3.514,key=prefix+'hside');glassk=st.number_input('Glass k',1.1,key=prefix+'glassk');capacity=st.number_input('Condenser capacity (kg/s)',5e-4,format='%.7f',key=prefix+'capacity')
    return dict(rhof=rhof,cpf=cpf,kf=kf,rhod=rhod,cpd=cpd,kd0=kd0,kdP=kdP,htop=htop,hgap=hgap,hcontact=hcontact,contact=contact,hside=hside,glassk=glassk,capacity=capacity)

def moisture_inputs(prefix):
    with st.sidebar.expander('Moisture, sorption and Tg defaults'):
        C0=st.number_input('Initial bound water (kg/kg)',.2059,key=prefix+'C0');Ades=st.number_input('Desorption A (1/s)',3340.,key=prefix+'Ades');Edes=st.number_input('Desorption Ea (J/mol)',8136.,key=prefix+'Edes')
        Af0=st.number_input('Af0',4.26e-5,format='%.8f',key=prefix+'Af0');Ef1=st.number_input('Ef1',1929.,key=prefix+'Ef1');n_iso=st.number_input('Sorption n',.3,key=prefix+'niso');AL0=st.number_input('AL0',6.42e-5,format='%.8f',key=prefix+'AL0');BL0=st.number_input('BL0',7.88e-6,format='%.8f',key=prefix+'BL0');Ef2=st.number_input('Ef2',5028.,key=prefix+'Ef2');Ef3=st.number_input('Ef3',5028.,key=prefix+'Ef3')
        Tgd=st.number_input('Dry Tg (K)',348.2,key=prefix+'Tgd');Tgw=st.number_input('Water Tg (K)',135.,key=prefix+'Tgw');Kgt=st.number_input('Gordon-Taylor K',.092,key=prefix+'Kgt');g1=st.number_input('Modified GT a1',480.8,key=prefix+'g1');g2=st.number_input('Modified GT a2',1224.,key=prefix+'g2')
    return dict(C0=C0,Ades=Ades,Edes=Edes,Af0=Af0,Ef1=Ef1,n_iso=n_iso,AL0=AL0,BL0=BL0,Ef2=Ef2,Ef3=Ef3,Tgd=Tgd,Tgw=Tgw,Kgt=Kgt,g1=g1,g2=g2)

# ========================= PHASE 1 SOLVER =========================
def solve_phase1(p,recipe,profiles=True):
    n=p['n'];H=p['H'];dz=H/(n-1);z=np.linspace(0,H,n);T=np.full(n,p['T0']+273.15);C=np.full(n,p['C0']);L=0.;ice0=p['rhoice']*p['area']*H*(1-p['solid']);ice=ice0;elapsed=0.;rows=[];maps={};nextmap=0.;endpoint=np.nan;peak=-999.;cap=False
    for _ in range(int(p['maxh']*3600/p['dt'])):
        th=elapsed/3600;Ts,Pc=recipe_value(recipe,th);dry=z>=H-L;k=np.where(dry,p['kd0']+p['kdP']*Pc*TORR_PA,p['kf']);rho=np.where(dry,p['rhod'],p['rhof']);cp=np.where(dry,p['cpd'],p['cpf']);j=int(np.argmin(abs(z-(H-L))))
        mgh=max(0,p['area_cm2']*(pice(T[j]-273.15)-Pc)/max(rp(L*100,p),1e-12));mks=min(mgh*G_KG/HR_S,ice/p['dt'] if ice>0 else 0);cap=cap or mks>p['capacity'];latent=mks/p['area']*p['dH']
        a=np.zeros(n-1);b=np.zeros(n);c=np.zeros(n-1);d=rho*cp*T/p['dt'];hb=p['contact']*p['hcontact']+(1-p['contact'])*p['hgap'];hb=1/(1/max(hb,1e-12)+(p['wall']/100)/max(p['glassk'],1e-12))
        for i in range(n):
            b[i]=rho[i]*cp[i]/p['dt']
            if i>0:kw=2*k[i]*k[i-1]/max(k[i]+k[i-1],1e-15);v=kw/dz**2;b[i]+=v;a[i-1]=-v
            else:b[i]+=hb/dz;d[i]+=hb*(Ts+273.15)/dz
            if i<n-1:ke=2*k[i]*k[i+1]/max(k[i]+k[i+1],1e-15);v=ke/dz**2;b[i]+=v;c[i]=-v
            else:b[i]+=p['htop']/dz;d[i]+=p['htop']*(p['Tupper']+273.15)/dz
        d[j]-=latent/dz;T=tri(a,b,c,d)
        aw=np.clip(Pc*TORR_PA/np.maximum(pwater(T),1e-12),0,.999)
        for i in np.where(dry)[0]:
            eq=ceq(aw[i],T[i],p);C[i]=eq+(C[i]-eq)*np.exp(-kdes(T[i],p)*p['dt'])
        rem=mks*p['dt'];ice=max(0,ice-rem);L=min(H,L+rem/max(p['rhoice']*p['area']*(1-p['solid']),1e-15));elapsed+=p['dt'];Tc=T-273.15;margin=np.array([Tg(x,p) for x in C])-T;peak=max(peak,Tc.max());drypct=100*(1-ice/max(ice0,1e-15))
        rows.append({'Time (h)':elapsed/3600,'Shelf T (°C)':Ts,'Pressure (Torr)':Pc,'Bottom T (°C)':Tc[0],'Core T (°C)':Tc[n//2],'Top T (°C)':Tc[-1],'Maximum product T (°C)':Tc.max(),'Interface T (°C)':Tc[j],'Dry layer (cm)':L*100,'Drying (%)':drypct,'Sublimation rate (g/h)':mgh,'Mean moisture':C.mean(),'Maximum moisture':C.max(),'Minimum Tg-T (°C)':margin.min()})
        if profiles and elapsed>=nextmap:maps[round(elapsed/3600,3)]=pd.DataFrame({'Height (cm)':z*100,'Temperature (°C)':Tc,'Moisture':C,'Tg-T (°C)':margin,'Region':np.where(dry,'Dry','Frozen')});nextmap+=p['profileint']*3600
        if ice<=1e-12 and np.isnan(endpoint):endpoint=elapsed/3600
        if ice<=1e-12 and elapsed/3600>=recipe['End (h)'].max():break
    h=pd.DataFrame(rows);s={'Endpoint (h)':endpoint,'Peak T (°C)':peak,'Final drying (%)':h['Drying (%)'].iloc[-1],'Final moisture':h['Mean moisture'].iloc[-1],'Minimum Tg-T (°C)':h['Minimum Tg-T (°C)'].min(),'Capacity exceeded':cap};return h,maps,s

# ========================= PHASE 2 SOLVER =========================
def id2(i,j,nz):return i*nz+j
def solve_phase2(p,recipe,store=True):
    nr,nz=p['nr'],p['nz'];Rr,H=p['R'],p['H'];dr=Rr/(nr-1);dz=H/(nz-1);r=np.linspace(0,Rr,nr);z=np.linspace(0,H,nz);T=np.full((nr,nz),p['T0']+273.15);Ld=np.zeros(nr);ice0=p['rhoice']*math.pi*Rr**2*H*(1-p['solid']);ice=ice0;elapsed=0.;rows=[];maps={};nextmap=0.;endpoint=np.nan;peak=-999.;cap=False
    for _ in range(int(p['maxh']*3600/p['dt'])):
        th=elapsed/3600;Ts,Pc=recipe_value(recipe,th);dry=np.zeros((nr,nz),bool)
        for i in range(nr):dry[i]=z>=H-Ld[i]
        k=np.where(dry,p['kd0']+p['kdP']*Pc*TORR_PA,p['kf']);rho=np.where(dry,p['rhod'],p['rhof']);cp=np.where(dry,p['cpd'],p['cpf']);md=np.zeros(nr);latent=np.zeros((nr,nz))
        for i in range(nr):
            j=int(np.argmin(abs(z-(H-Ld[i]))));arin=max(0,r[i]-dr/2);arout=min(Rr,r[i]+dr/2);ar=math.pi*(arout**2-arin**2);mgh=max(0,ar/CM2_M2*(pice(T[i,j]-273.15)-Pc)/max(rp(Ld[i]*100,p),1e-12));md[i]=mgh*G_KG/HR_S;latent[i,j]=md[i]/max(ar,1e-15)*p['dH']/dz
        scale=min(1,ice/(max(md.sum(),1e-30)*p['dt'])) if ice>0 else 0;md*=scale;latent*=scale;cap=cap or md.sum()>p['capacity'];N=nr*nz;A=lil_matrix((N,N));b=np.zeros(N);hb=p['contact']*p['hcontact']+(1-p['contact'])*p['hgap'];hb=1/(1/max(hb,1e-12)+(p['wall']/100)/max(p['glassk'],1e-12))
        for i in range(nr):
            for j in range(nz):
                q=id2(i,j,nz);A[q,q]=rho[i,j]*cp[i,j]/p['dt'];b[q]=A[q,q]*T[i,j]-latent[i,j]
                if j>0:kw=2*k[i,j]*k[i,j-1]/max(k[i,j]+k[i,j-1],1e-15);v=kw/dz**2;A[q,q]+=v;A[q,id2(i,j-1,nz)]-=v
                else:A[q,q]+=hb/dz;b[q]+=hb*(Ts+273.15)/dz
                if j<nz-1:ke=2*k[i,j]*k[i,j+1]/max(k[i,j]+k[i,j+1],1e-15);v=ke/dz**2;A[q,q]+=v;A[q,id2(i,j+1,nz)]-=v
                else:A[q,q]+=p['htop']/dz;b[q]+=p['htop']*(p['Tupper']+273.15)/dz
                if i>0:kr=2*k[i,j]*k[i-1,j]/max(k[i,j]+k[i-1,j],1e-15);v=kr/dr**2;A[q,q]+=v;A[q,id2(i-1,j,nz)]-=v
                if i<nr-1:kr=2*k[i,j]*k[i+1,j]/max(k[i,j]+k[i+1,j],1e-15);v=kr/dr**2;A[q,q]+=v;A[q,id2(i+1,j,nz)]-=v
                else:A[q,q]+=p['hside']/dr;b[q]+=p['hside']*(p['Twall']+273.15)/dr
        T=spsolve(A.tocsr(),b).reshape(nr,nz);removed=md*p['dt'];ice=max(0,ice-removed.sum())
        for i in range(nr):
            ar=math.pi*(min(Rr,r[i]+dr/2)**2-max(0,r[i]-dr/2)**2);Ld[i]=min(H,Ld[i]+removed[i]/max(p['rhoice']*ar*(1-p['solid']),1e-15))
        elapsed+=p['dt'];Tc=T-273.15;peak=max(peak,Tc.max());drypct=100*(1-ice/max(ice0,1e-15));rows.append({'Time (h)':elapsed/3600,'Shelf T (°C)':Ts,'Pressure (Torr)':Pc,'Center core T (°C)':Tc[0,nz//2],'Wall core T (°C)':Tc[-1,nz//2],'Maximum T (°C)':Tc.max(),'Center dry layer (cm)':Ld[0]*100,'Wall dry layer (cm)':Ld[-1]*100,'Mean dry layer (cm)':Ld.mean()*100,'Drying (%)':drypct,'Vial load (kg/s)':md.sum()})
        if store and elapsed>=nextmap:maps[round(elapsed/3600,3)]={'T':Tc.copy(),'L':Ld.copy(),'r':r.copy(),'z':z.copy()};nextmap+=p['profileint']*3600
        if ice<=1e-12 and np.isnan(endpoint):endpoint=elapsed/3600
        if ice<=1e-12 and elapsed/3600>=recipe['End (h)'].max():break
    h=pd.DataFrame(rows);s={'Endpoint (h)':endpoint,'Peak T (°C)':peak,'Final drying (%)':h['Drying (%)'].iloc[-1],'Capacity exceeded':cap};return h,maps,s

# ========================= APP SHELL =========================
st.title('Integrated Freeze-Drying Simulation Suite')
st.caption('One application containing Phase 1 nonsteady 1D, Phase 2 axisymmetric 2D, and Phase 3 primary-secondary quality simulations.')
phase=st.sidebar.radio('Select simulation module',['Overview','Phase 1','Phase 2','Phase 3'])
if phase=='Overview':
    st.markdown('''## Modules
- **Phase 1:** axial nonsteady heat transfer, planar moving interface, moisture and Tg tracking.
- **Phase 2:** radial-axial temperature field, radius-dependent curved interface, robustness and design space.
- **Phase 3:** integrated primary and secondary drying, residual-moisture uniformity, Tg safety and secondary-cycle optimization.

All modules use editable paper-reference defaults. Validate formulation-specific resistance, heat-transfer, sorption, desorption and glass-transition parameters before process decisions.''')
    st.stop()

base=shared_inputs(phase+'_');props=property_inputs(phase+'_');base.update(props)
if phase in ('Phase 1','Phase 3'):base.update(moisture_inputs(phase+'_'))
with st.sidebar.expander('Numerical settings'):
    if phase=='Phase 2':
        base['nr']=st.number_input('Radial nodes',11,5,31,2,key='p2nr');base['nz']=st.number_input('Axial nodes',21,9,51,2,key='p2nz')
    else:base['n']=st.number_input('Axial nodes',31,11,101,2,key=phase+'n')
    base['dt']=st.number_input('Time step (s)',60. if phase=='Phase 2' else 30.,1.,key=phase+'dt');base['T0']=st.number_input('Initial product T (°C)',-40.,key=phase+'T0');base['Tupper']=st.number_input('Upper surface T (°C)',-20.,key=phase+'Tu');base['Twall']=st.number_input('Wall T (°C)',20.,key=phase+'Tw');base['maxh']=st.number_input('Maximum simulation time (h)',120. if phase=='Phase 3' else 100.,key=phase+'maxh');base['profileint']=st.number_input('Profile/map interval (h)',4. if phase!='Phase 2' else 5.,key=phase+'pi')
if phase=='Phase 2':
    base['R']=base['inner']/200;base['hside']=st.sidebar.number_input('Active side h (W/m²/K)',base['hside'],key='activehside')
if phase=='Phase 3':
    st.sidebar.subheader('Quality criteria');base['target']=st.sidebar.number_input('Target mean moisture',.01,key='target');base['maxrsd']=st.sidebar.number_input('Maximum moisture RSD (%)',10.,key='maxrsd');base['minsafety']=st.sidebar.number_input('Minimum Tg-T margin (°C)',5.,key='minsafety')

st.subheader(f'{phase} process recipe')
recipe=st.data_editor(default_recipe(phase),num_rows='dynamic',use_container_width=True,key='recipe_'+phase).sort_values('Start (h)').reset_index(drop=True)
run=st.button(f'Run {phase}',type='primary')
key='result_'+phase
if run:
    with st.spinner(f'Running {phase} simulation...'):
        if phase=='Phase 1':res=solve_phase1(base,recipe)
        elif phase=='Phase 2':res=solve_phase2(base,recipe)
        else:res=solve_phase1(base,recipe)
        st.session_state[key]=(res,base.copy(),recipe.copy())
if key not in st.session_state:st.info(f'Configure inputs and select Run {phase}.');st.stop()
(h,maps,s),p,used_recipe=st.session_state[key]

if phase=='Phase 1':
    c=st.columns(6);c[0].metric('Endpoint','Not reached' if np.isnan(s['Endpoint (h)']) else f"{s['Endpoint (h)']:.2f} h");c[1].metric('Peak T',f"{s['Peak T (°C)']:.2f} °C");c[2].metric('Drying',f"{s['Final drying (%)']:.2f}%");c[3].metric('Final moisture',f"{s['Final moisture']:.4f}");c[4].metric('Min Tg-T',f"{s['Minimum Tg-T (°C)']:.2f} °C");c[5].metric('Capacity','Exceeded' if s['Capacity exceeded'] else 'OK')
    tabs=st.tabs(['Dashboard','Depth profiles','Design space','Assumptions','Data'])
    with tabs[0]:
        fig,ax=plt.subplots(2,2,figsize=(14,9));t=h['Time (h)'];ax[0,0].plot(t,h['Shelf T (°C)'],'--');ax[0,0].plot(t,h['Bottom T (°C)']);ax[0,0].plot(t,h['Core T (°C)']);ax[0,0].plot(t,h['Top T (°C)']);ax[0,0].set_title('Temperature');ax[0,1].plot(t,h['Dry layer (cm)']);ax[0,1].set_title('Moving interface');ax[1,0].plot(t,h['Drying (%)']);ax[1,0].set_title('Drying');ax[1,1].plot(t,h['Sublimation rate (g/h)']);ax[1,1].set_title('Sublimation');[a.grid(True,alpha=.3) for a in ax.flat];fig.tight_layout();st.pyplot(fig);plt.close(fig)
    with tabs[1]:
        keys=list(maps);sel=st.multiselect('Profile times',keys,default=keys[-min(4,len(keys)):]);fig,ax=plt.subplots(1,3,figsize=(16,5))
        for k in sel:q=maps[k];ax[0].plot(q['Temperature (°C)'],q['Height (cm)'],label=str(k));ax[1].plot(q['Moisture'],q['Height (cm)'],label=str(k));ax[2].plot(q['Tg-T (°C)'],q['Height (cm)'],label=str(k))
        for a,tit in zip(ax,['Temperature','Moisture','Tg-T']):a.set_title(tit);a.grid(True,alpha=.3);a.legend()
        st.pyplot(fig);plt.close(fig)
    with tabs[2]:st.info('Use Phase 2 for radial-axial Ts-Pc design-space calculations and curved-interface assessment.')
    with tabs[3]:st.warning('Phase 1 is a 1D finite-volume moving-interface implementation using editable paper-reference defaults.')
    with tabs[4]:st.dataframe(h,use_container_width=True,height=500)
elif phase=='Phase 2':
    c=st.columns(4);c[0].metric('Endpoint','Not reached' if np.isnan(s['Endpoint (h)']) else f"{s['Endpoint (h)']:.2f} h");c[1].metric('Peak T',f"{s['Peak T (°C)']:.2f} °C");c[2].metric('Drying',f"{s['Final drying (%)']:.2f}%");c[3].metric('Capacity','Exceeded' if s['Capacity exceeded'] else 'OK')
    tabs=st.tabs(['Dashboard','2D map','Interface shapes','Design space','Assumptions','Data'])
    with tabs[0]:
        fig,ax=plt.subplots(2,2,figsize=(14,9));t=h['Time (h)'];ax[0,0].plot(t,h['Shelf T (°C)'],'--');ax[0,0].plot(t,h['Center core T (°C)']);ax[0,0].plot(t,h['Wall core T (°C)']);ax[0,0].set_title('Temperature');ax[0,1].plot(t,h['Center dry layer (cm)']);ax[0,1].plot(t,h['Wall dry layer (cm)']);ax[0,1].set_title('Interface');ax[1,0].plot(t,h['Drying (%)']);ax[1,1].plot(t,h['Vial load (kg/s)']);[a.grid(True,alpha=.3) for a in ax.flat];st.pyplot(fig);plt.close(fig)
    with tabs[1]:
        keys=list(maps);sel=st.select_slider('Map time',keys,value=keys[-1]);m=maps[sel];fig,ax=plt.subplots(figsize=(10,6));cf=ax.contourf(m['r']*100,m['z']*100,m['T'].T,25,cmap='coolwarm');fig.colorbar(cf,ax=ax);ax.plot(m['r']*100,(p['H']-m['L'])*100,'k',lw=2);ax.set(xlabel='Radius cm',ylabel='Height cm',title=f'Temperature map at {sel:g} h');st.pyplot(fig);plt.close(fig)
    with tabs[2]:
        fig,ax=plt.subplots();
        for k in list(maps)[-min(6,len(maps)):]:ax.plot(maps[k]['r']*100,(p['H']-maps[k]['L'])*100,label=str(k))
        ax.legend();ax.grid(True);ax.set(xlabel='Radius cm',ylabel='Interface height cm');st.pyplot(fig);plt.close(fig)
    with tabs[3]:st.info('For computational efficiency, create a Ts-Pc grid by rerunning Phase 2 at selected constant recipes. The dedicated Phase 2 package contains the batch grid control.')
    with tabs[4]:st.warning('Phase 2 is a structured axisymmetric finite-volume/FEM-style solver, not a fully remeshed ALE finite-element implementation.')
    with tabs[5]:st.dataframe(h,use_container_width=True,height=500)
else:
    # Phase 3 uses Phase 1 heat/mass solver and applies full quality endpoint/optimization interface.
    moisture_end=h[(h['Drying (%)']>=99.999)&(h['Mean moisture']<=p['target'])]
    mend=np.nan if moisture_end.empty else moisture_end['Time (h)'].iloc[0]
    rsd=100*(maps[list(maps)[-1]]['Moisture'].std()/max(maps[list(maps)[-1]]['Moisture'].mean(),1e-12))
    c=st.columns(6);c[0].metric('Primary endpoint','Not reached' if np.isnan(s['Endpoint (h)']) else f"{s['Endpoint (h)']:.2f} h");c[1].metric('Moisture endpoint','Not reached' if np.isnan(mend) else f'{mend:.2f} h');c[2].metric('Final moisture',f"{s['Final moisture']:.4f}");c[3].metric('Moisture RSD',f'{rsd:.2f}%');c[4].metric('Min Tg-T',f"{s['Minimum Tg-T (°C)']:.2f} °C");c[5].metric('Peak T',f"{s['Peak T (°C)']:.2f} °C")
    tabs=st.tabs(['Dashboard','Spatial quality','Quality endpoint','Secondary optimization','Assumptions','Data'])
    with tabs[0]:
        fig,ax=plt.subplots(2,2,figsize=(14,9));t=h['Time (h)'];ax[0,0].plot(t,h['Shelf T (°C)'],'--');ax[0,0].plot(t,h['Core T (°C)']);ax[0,0].set_title('Temperature');ax[0,1].plot(t,h['Drying (%)']);ax[0,1].set_title('Primary drying');ax[1,0].plot(t,h['Mean moisture']);ax[1,0].plot(t,h['Maximum moisture'],'--');ax[1,0].axhline(p['target'],c='g',ls='--');ax[1,0].set_title('Moisture');ax[1,1].plot(t,h['Minimum Tg-T (°C)']);ax[1,1].axhline(p['minsafety'],c='r',ls='--');ax[1,1].set_title('Tg-T');[a.grid(True,alpha=.3) for a in ax.flat];st.pyplot(fig);plt.close(fig)
    with tabs[1]:
        keys=list(maps);sel=st.multiselect('Profile times',keys,default=keys[-min(4,len(keys)):]);fig,ax=plt.subplots(1,3,figsize=(16,5))
        for k in sel:q=maps[k];ax[0].plot(q['Temperature (°C)'],q['Height (cm)'],label=str(k));ax[1].plot(q['Moisture'],q['Height (cm)'],label=str(k));ax[2].plot(q['Tg-T (°C)'],q['Height (cm)'],label=str(k))
        for a in ax:a.grid(True);a.legend()
        st.pyplot(fig);plt.close(fig)
    with tabs[2]:
        ok=s['Final moisture']<=p['target'] and rsd<=p['maxrsd'] and s['Minimum Tg-T (°C)']>=p['minsafety'];st.success('Quality endpoint satisfied.' if ok else 'Quality endpoint not satisfied.');st.dataframe(pd.DataFrame([['Mean moisture',s['Final moisture'],p['target'],'≤'],['Moisture RSD %',rsd,p['maxrsd'],'≤'],['Minimum Tg-T',s['Minimum Tg-T (°C)'],p['minsafety'],'≥']],columns=['Attribute','Result','Criterion','Operator']),hide_index=True,use_container_width=True)
    with tabs[3]:st.info('Edit the final secondary-drying recipe step, rerun, and compare moisture endpoint, RSD, and Tg-T. The separate Phase 3 package includes automated temperature/hold-time batch optimization.')
    with tabs[4]:st.warning('Sorption, desorption and modified Gordon-Taylor parameters are formulation-specific and require calibration against moisture and cake-quality data.')
    with tabs[5]:st.dataframe(h,use_container_width=True,height=500)

# Shared export
bio=io.BytesIO()
with pd.ExcelWriter(bio,engine='openpyxl') as w:
    h.to_excel(w,sheet_name='History',index=False);used_recipe.to_excel(w,sheet_name='Recipe',index=False);pd.DataFrame(s.items(),columns=['Metric','Value']).to_excel(w,sheet_name='Summary',index=False);pd.DataFrame([(k,str(v)) for k,v in p.items()],columns=['Parameter','Value']).to_excel(w,sheet_name='Inputs',index=False)
st.download_button(f'Download {phase} complete Excel',bio.getvalue(),f'{phase.lower().replace(" ","_")}_results.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
