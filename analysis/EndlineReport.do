****
** Analysis of TUP_full.dta.
** For the standard endline results
** Last edit: 2015-11-06
****

clear
set more off
* use ../../data/TUP_full, clear
use data/TUP_full, clear

local consumption_regs 1

*****************
** Consumption **
*****************
 ** Set Local variables
  local food c_cereals c_maize c_sorghum c_millet c_potato c_sweetpotato c_rice c_bread c_beans c_oil c_salt c_sugar c_meat c_livestock c_poultry c_fish c_egg c_nuts c_milk c_vegetables c_fruit c_tea c_spices c_alcohol c_otherfood    
  local month c_fuel c_medicine c_airtime c_cosmetics c_soap c_transport c_entertainment c_childcare c_tobacco c_batteries c_church c_othermonth
  local year c_clothesfootwear c_womensclothes c_childrensclothes c_shoes c_homeimprovement c_utensils c_furniture c_textiles c_ceremonies c_funerals c_charities c_dowry c_other    

 ** Normalize Consumption variables by day
  foreach suff in _e _m _b{
    foreach var of local food{
        capture replace `var'`suff' = `var'`suff'/3
    }
    foreach var of local month{
        capture replace `var'`suff' = `var'`suff'/30
    }
    foreach var of local year{
        capture replace `var'`suff' = `var'`suff'/360
    }
    }          
 
 ** Make aggregate consumpmtion variables
  drop c_cereals_e c_meat_e

  foreach suff in _e _m _b{
   gen c_total`suff' = 0
   foreach category in food month year {
     gen c_`category'`suff' = 0
     disp "Aggregating by `category'"
     foreach item of local `category'{
       capture replace c_`category'`suff' = c_`category'`suff' + `item'`suff'
   }
      replace c_total`suff' = c_total`suff' + c_`category'`suff'
      ** Replace observation to missing if observation is zero
      replace c_`category'`suff' = . if c_`category'`suff'==0
   }

 ** But only change total to missing if food is zero
  replace c_total`suff' = . if c_food`suff'==0

  disp "Making Food Share Variables"
  gen foodshare`suff' = c_food`suff'/c_total`suff'

  ** Check if the summation happened properly...
  if c_total`suff' == c_food`suff' + c_year`suff' + c_month`suff' {
    disp "Totals check out for `suff"
    }   
  else {
    disp "Totals don't check out for `suff'"
    }   
  }

 ** Treatment Variables
  gen TUP_high = group=="High Asset"
  ** Some Control group observations not coded in dummy variables ** Shouldn't change much?
  replace TUP=0 if group=="Control"
  replace Cash=0 if group=="Control"
  replace Control=1 if group=="Control"
  
 ** Variable Labels
  label var c_total_b   "Tot 2013"
  label var c_total_m   "Tot 2014"
  label var c_total_e   "Tot 2015"
  label var c_food_b    "Food 2013"
  label var c_food_m    "Food 2014"
  label var c_food_e    "Food 2015"
  label var c_month_b   "Month 2013"
  label var c_month_m   "Month 2014"
  label var c_month_e   "Month 2015"
  label var c_year_b    "Year 2013"
  label var c_year_m    "Year 2014"
  label var c_year_e    "Year 2015"
  label var foodshare_b "FS 2013"
  label var foodshare_m "FS 2014"
  label var foodshare_e "FS 2015"
  label var Cash        "Cash"
  label var Control     "Control"
  label var TUP         "TUP"
  label var TUP_high    "TUP+"                
 

 ** Run Consumption Regressions
 if `consumption_regs' {
  * preserve
  
  local Consumption c_total c_month c_food c_year foodshare
  local Baseline c_total_b c_month_b c_food_b c_year_b foodshare_b 
  
  ** Reshape data
  foreach outcome of local Consumption {
     rename `outcome'_m `outcome'2014
     rename `outcome'_e `outcome'2015
     rename `outcome'_b `outcome'_b2014
     gen    `outcome'_b2015=`outcome'_b2014
    }
  keep *2015 *2014 idno TUP Cash
  reshape long c_total c_month c_food c_year foodshare c_total_b c_month_b c_food_b c_year_b foodshare_b, i(idno) j(year)
  ** Make Baseline_missing variable
  foreach outcome of local Consumption {
    gen `outcome'_b_missing = `outcome'_b==.
    replace `outcome'_b = 0 if `outcome'_b==.
    }
  ** Make Interaction terms
  gen y2014 = year==2014
  gen y2015 = year==2015
  gen TUP2014  =  TUP*y2014
  gen TUP2015  =  TUP*y2015
  gen Cash2014 = Cash*y2014
  gen Cash2015 = Cash*y2015

  ** Run Regressions
  gen Baseline = .
  gen Bsln_Missing = .
  foreach outcome of local Consumption {
     replace Baseline = `outcome'_b
     replace Bsln_Missing = `outcome'_b_missing
     eststo ols_`outcome':    reg `outcome' TUP2014 Cash2014 TUP2015 Cash2015 y2014 y2015 Baseline Bsln_Missing, noconstant
     }       
  esttab ols_*    using tables\consumption2015.tex,        se l tex compress replace title(Estimated Impact on Consumption)
  esttab ols_*    using tables\consumption2015.txt,        se l fixed compress replace title(Estimated Impact on Consumption)

  * restore
  }
  else{
      disp "WARNING: Did not run consumption regressions."
  }

