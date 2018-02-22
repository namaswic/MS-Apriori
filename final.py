import numpy as np

import pandas as pd

import itertools
from numpy import nan


trans=pd.read_table("transaction-s.txt",header=None)########## Enter transaction file here 
trans=trans.iloc[0:, 0].str.rsplit(' ', expand=True)

trans.replace(to_replace="{",value="",regex=True,inplace=True)
trans.replace(to_replace=",",value="",regex=True,inplace=True)
trans.replace(to_replace="}",value="",regex=True,inplace=True)
trans.fillna(value=nan, inplace=True)
trans1=trans.iloc[:,:]
trans_numpya=trans.iloc[:,:]
trans_numpya=np.asarray(trans_numpya)

para=pd.read_fwf("parameter-s.txt",header=None,delimiter ="=")# Enter parameter file here

mis_set=para[0:-3]
mis_set=mis_set.iloc[:, 0].str.rsplit('=', expand=True)
mis_set.replace(to_replace='[()]',value="",regex=True,inplace=True)
mis_set.replace(to_replace="MIS",value="",regex=True,inplace=True)
mis_set.replace(to_replace=" ",value="",regex=True,inplace=True)
mis_set.columns=['item','MIS']

mis_set1=mis_set.sort_values(['MIS'])

sdc_data=para.iloc[-3, 0:1].str.rsplit('=', expand=True)
sdc_data.columns=['sdc','value']
sdc=sdc_data.value[0]



cbt=para.iloc[-2, 0:].str.split(':', expand=True)
cbt=cbt.iloc[0, 0:].str.split('{[0-9],[0-9]},', expand=True)
cbt=cbt.iloc[0:, 0].str.split('}', expand=True)
cbt.replace(to_replace='[A-z]',value="",regex=True,inplace=True)
cbt=cbt.iloc[1:,0:]
cbt.replace(to_replace='{',value="",regex=True,inplace=True)
cbt=cbt.transpose()
cbt.replace(to_replace=',',value="",regex=True,inplace=True)
cbt=cbt.iloc[0:, 0].str.split(' ', expand=True)  
cbt=cbt.iloc[0:,1:]  
cbt=cbt.iloc[0:-1,0:] 
cbt=np.asarray(cbt)   



mh=para.iloc[-1, 0:].str.rsplit('or', expand=True)
mh.replace(to_replace='[A-z]',value="",regex=True,inplace=True)
mh.replace(to_replace='-:',value="",regex=True,inplace=True)
mh.replace(to_replace=' ',value="",regex=True,inplace=True)

mh=mh.transpose()
mh=np.asarray(mh)
mh=list(mh)

no_of_trans=len(trans.index)

trans_list=[]

dict1={}
dict_tail={}
for i in range(len(trans.columns)):
        for j in trans[i]:
            trans_list.append(j)


def ms_apriori( trans, mis_set, sdc):
  
   L = init_pass(mis_set1,trans)
   
   F1=check(L,mis_set)
   
   
   
   MIS_for_items_F1=[]
   for item in F1:
        
        MIS_for_items_F1.append(get_MISvalue(item))
    
    
   temp_data=pd.DataFrame(
    {'item': F1,
     'mis': MIS_for_items_F1
    })
   temp_data=temp_data.sort_values(by=['mis'])
   F1 = temp_data['item'].tolist()
   F1=np.asarray(F1)
   
   F=F1
   F_last=[]
   k=2
   
   while (len(F1)!=0):
       
       if k==2:
           C=level2_candiate_gen(L,sdc)
           C=np.asarray(C)
           
           
       else:
           
           
           C=MScandiate_gen(F1,sdc,k)
           C=np.asarray(C)
           
           
       countfor_loop=0
       
       
       counts_cands=[0]*len(C)
       counts_cands_wf=[0]*len(C)
       for it, t in enumerate(trans_numpya):
          for ic, c in enumerate(C):
              
               if set(c).issubset(set(t)):
                  
                  countfor_loop=countfor_loop+1
                  
                  counts_cands[ic]=counts_cands[ic]+1
                  
                  
               if set(np.delete(c, 0)).issubset(set(t)):
                  counts_cands_wf[ic]=counts_cands_wf[ic]+1
       for ic, c in enumerate(C):
          
           dict1.update({np.array_str(c): counts_cands[ic]})
           
           dict_tail.update({np.array_str(c): counts_cands_wf[ic]})
           
       
       if k!=2:
        for i,e in enumerate(F1):
         
         F_last.append(e)
       F1=[]
      
       for ic, c in enumerate(C):
           
            if pd.to_numeric(counts_cands[ic]/float(no_of_trans)) >= pd.to_numeric(get_MISvalue(c[0])):
               
             
               F1.append(c)
      
       
       k=k+1
   return F_last,F




def is_subset(a, b):
  b = np.unique(b)
  c = np.intersect1d(a,b)
  return c.size == b.size


def MScandiate_gen(F1,sdc,k):
    
    CMS=[]
    
    remove_list=[]
    for i1, f1 in enumerate(F1):
     for i2,f2 in enumerate(F1):
       
        if(checkforsim(f1,f2)): 
            
             if(pd.to_numeric(F1[i1][k-2])<pd.to_numeric(F1[i2][k-2])):
                 
                 val1=(trans_list.count(str(pd.to_numeric(F1[i1][k-2]))))/no_of_trans
                 val2=(trans_list.count(str(pd.to_numeric(F1[i2][k-2]))))/no_of_trans
                
                 if(abs(val1-val2)<=sdc):
                    
                     if(pd.to_numeric(get_MISvalue(F1[i1][k-2]))<=pd.to_numeric(get_MISvalue(F1[i2][k-2]))):
                      c=np.append(f1,F1[i2][k-2])
                     else:
                       c=np.append(f2,F1[i1][k-2])
                    
                     CMS.append(list(c))
                     
                     for s in list(itertools.combinations(c,k-1))  :
                          mark=False
                          
                          if ((c[0] in s) or (pd.to_numeric(get_MISvalue(c[1]))==pd.to_numeric(get_MISvalue(c[0])))):
                              
                              if(findifsubsettoremove(s,F1)):
                                 
                                  remove_list.append(list(c))
                                  
                                  
                                  mark=True
                          if(mark==True):
                            
                             break
                           
    temp=[]  
        
    if(len(remove_list)!=0) :  
    
     
         for c in CMS:
          if c not in remove_list:
              temp.append(c)
        
    else:
        temp=CMS
    
   
    return temp

def checkforsim(f1,f2) :
    length=len(f1)
   
    for i in range(length-1):
        if(f1[i]!=f2[i]):
          
            return False
    return True
    
    
    
    
def findifsubsettoremove(s,F1):
    for items in F1:
        b=True
        for i,j in zip(items,s):
            
            if(pd.to_numeric(i)!=pd.to_numeric(j)):
                
                b=False
        if(b==True):
       
            return False
    return True
        

    
    
    
                                           
         
def check(L,sdc):
    trans_list=[]
    item_list=[]
    
    l1=[]
    
    for item in L:
        item_list.append(item)
    
    
    for i in range(len(trans.columns)):
        for j in trans[i]:
            trans_list.append(j)
    
    
    for item in range(len(item_list)):
        count=0
       
        count=trans_list.count(str(pd.to_numeric(item_list[item])))
       
        if((count/float(len(trans.index))) >= pd.to_numeric(get_MISvalue(item_list[item]))):
            
            l1.append(str(pd.to_numeric(item_list[item])))
          
    return np.asarray(l1)

def get_MISvalue(item):
    mis=[]
    for item1 in mis_set['MIS']:
        mis.append(item1)
    index= mis_set[mis_set['item'] == item].index.tolist()
    
    return mis[index[0]]


def level2_candiate_gen(L,sdc):
    C2=[]
    l=[]
    
    items=[]
    trans_list=[]
    temp_data=[]
    for item in L:
        l.append(item)
    
    MIS_for_items=[]
    for item in L:
       
        MIS_for_items.append(get_MISvalue(item))
    
    
    temp_data=pd.DataFrame(
    {'item': l,
     'mis': MIS_for_items 
    })
    temp_data=temp_data.sort_values(by=['mis'])
    items = temp_data['item'].tolist()

    for i in range(len(trans.columns)):
        for j in trans[i]:
            trans_list.append(j)
    for idx, val in enumerate(items):
          
            count=0
       
            count=trans_list.count(str(pd.to_numeric(val)))
            
            if((count/float(len(trans.index))) >= pd.to_numeric(get_MISvalue(val))):
                
                for idx1, val1 in enumerate(items):
                   list1=[]
                 
                   if(idx1>idx):
                       val1count=trans_list.count(str(pd.to_numeric(val1)))/float(len(trans.index))
                       valcount=trans_list.count(str(pd.to_numeric(val)))/float(len(trans.index))
                       
                       
                       if(( val1count >= pd.to_numeric(get_MISvalue(val))) and (abs(val1count-valcount))<=pd.to_numeric(sdc)):
                        
                          list1= [val,val1]
                          C2.append(list1)
                           
    return C2                 

                
    
def getloc(item,L):
    
    l=[]
    for item1 in L:
        l.append(item1)
    for i in range(len(L)):
        if(L[i]==item):
           
            position=i
            break
    return position
        
    


def init_pass(mis_set,trans):
    trans_list=[]
    item_list=[]
    mis=[]
    l=[]
    no_of_trans=len(trans.index)
    for item in mis_set1['item']:
        item_list.append(item)
    for item in mis_set1['MIS']:
        mis.append(item)
    for i in range(len(trans.columns)):
        for j in trans[i]:
            trans_list.append(j)
    
    
    for item in range(len(item_list)):
        count=0
       
        count=trans_list.count(str(pd.to_numeric(item_list[item])))
       
        if((count/float(no_of_trans)) >= pd.to_numeric(mis[item])):
            l.append(item_list[item])
            position=item
            break
            
    for item in range(len(item_list)):
        count=0
        
        if(pd.to_numeric(item_list[item])!=pd.to_numeric(l[0])):
           count=trans_list.count(str(pd.to_numeric(item_list[item])))
        
           if((count/float(no_of_trans)) >=pd.to_numeric( mis[position])):
            
            l.append(item_list[item])
    
    l=set(l)
   

    return l

def result():
    final=[]
    final,F1_items=ms_apriori( trans, mis_set, sdc)
    
    final_cbt=[]
   
    print "Frequent 1-Itemsets"
    print ""
    count=0

    for cand in F1_items:
        for item in mh:
            booll=False
            
            if cand==item :
              count=count+1
              print trans_list.count(str(pd.to_numeric(cand)))," : {",cand,"}"
              break
    print ""
    print "    Total number of frequent 1-itemsets = ",count
    print ""   
    
    for cand in final:
        booll=False
        for item in cbt:
           
            if set(item).issubset(set(cand)):
                booll=True
        if booll==False:
             final_cbt.append(cand)
        
    next=3
    k=2
    printed=False
    print "Frequent", k,"-Itemsets"
    print ""
    count1=0
    list(final_cbt)
    for cand in final_cbt:
        booll=False
        if(len(cand)==next and k==next-1):
            k=k+1
            next=next+1
            print "    Total number of frequent", (k-1),"-itemsets = ",count1
            count1=0
            print ""
            print "Frequent", k,"-Itemsets"
            print ""
            printed==True
            
            
        for item in mh:
            
            
           
            if set(item).issubset(set(cand)):
               
                booll=True
        if booll==True:
              print "    ",dict1.get(np.array_str(cand))," : {",', '.join(map(str, cand)),"}"
              print "TailCount:",dict_tail.get(np.array_str(cand))
              print ""
              count1=count1+1
            
            
    print "    Total number of frequent", k,"-itemsets = ",count1
      
    print ""

result()