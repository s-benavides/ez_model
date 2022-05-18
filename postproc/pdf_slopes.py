import scipy.odr as odr
import numpy as np
import pdf_ccdf

def pdf_slopes(exp_list,taus,dtaus,fluxes,datmin,multf,nbin=55,dqy=10):
    
    ## First calculate the PDFs from exp_list and data. Gives taus, pdf tail slopes and errors.
    taus_p = []
    dtaus_p = []
    ms = []
    ms_old = []
    mspl = dict([])
    for ii,run in enumerate(exp_list):
        # Create PDF
        dat = fluxes[run.encode('utf-8')][1]
        pdf_q_wts, x_q_wts = pdf_ccdf.pdf_ccdf(dat[dat>0], n=nbin,xmin = datmin,xmax = max(dat), output='pdf')

        # Fit PDF tail
        m,b = np.polyfit(np.log10(x_q_wts[np.where((x_q_wts>datmin) & (x_q_wts<multf*datmin))]),np.log10(pdf_q_wts[np.where((x_q_wts>datmin) & (x_q_wts<multf*datmin))]),1)
        ms.append(m)
        taus_p.append(taus[run])
        dtaus_p.append(dtaus[run])
        mspl[run] = m
        
    ## Now calculate the error

    # Starting points:
    if 'ng' in exp_list:
        fitmax = 9e-2
        print("Natural grains, fitting to 6e-2.")
    else:
        fitmax = 1e-2
    bqs = np.linspace(datmin,fitmax,5)

    mse = dict([])
    for bq in bqs:
        eq = bq*dqy
        ms2 = dict([])
        for ii,run in enumerate(exp_list):
            dat = fluxes[run.encode('utf-8')][1]
            pdf_q_wts, x_q_wts = pdf_ccdf.pdf_ccdf(dat, n=nbin,xmin = datmin,xmax = max(dat), output='pdf')
            m,b = np.polyfit(np.log10(x_q_wts[np.where((x_q_wts>bq) & (x_q_wts<eq))]),np.log10(pdf_q_wts[np.where((x_q_wts>bq) & (x_q_wts<eq))]),1)
            ms2[run]=m
        mse[bq]=ms2
        
    errsl = dict([])
    for exp in exp_list:
        minsl = min(np.array([mse[bqs[i]][exp] for i in range(len(bqs))]))
        maxsl = max(np.array([mse[bqs[i]][exp] for i in range(len(bqs))]))
        errsl[exp]=abs(mspl[exp]-np.array([minsl,maxsl]))

    # First put plotting errors in appropriate form:
    errsl_p = []
    for run in exp_list:
        errsl_p.append(np.max(errsl[run]))
        
    ## Fitting
    # Linear fit
    def func(B, x): return B[0]*x+B[1]
    linear = odr.Model(func)
    mydata = odr.RealData(taus_p,ms,sx=dtaus_p,sy=errsl_p)
    myodr=odr.ODR(mydata,linear,beta0=[10.,-1.5])
    myoutput=myodr.run()
    m1 = myoutput.beta[0]
    dm1 =myoutput.sd_beta[0]
    b1 =  myoutput.beta[1]
    db1 = myoutput.sd_beta[1]

    # Calculation of S and tau_c
    S = 1/m1
    tau_c = (-1-b1)/m1
    dS = S * np.sqrt((dm1/m1)**2)
    dtau_c = tau_c * np.sqrt( (dm1/m1)**2 + (db1/b1)**2)
    
    return [b1,m1,mspl,errsl,tau_c,dtau_c,S,dS]