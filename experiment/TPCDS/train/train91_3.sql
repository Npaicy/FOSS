-- start query 1 in stream 2 using template query91.tpl
select  
        cc_call_center_id Call_Center,
        cc_name Call_Center_Name,
        cc_manager Manager,
        sum(cr_net_loss) Returns_Loss
from
        call_center,
        catalog_returns,
        date_dim,
        customer,
        customer_address,
        customer_demographics,
        household_demographics
where
        catalog_returns.cr_call_center_sk       = call_center.cc_call_center_sk
and     catalog_returns.cr_returned_date_sk     = date_dim.d_date_sk
and     catalog_returns.cr_returning_customer_sk= customer.c_customer_sk
and     customer_demographics.cd_demo_sk              = customer.c_current_cdemo_sk
and     household_demographics.hd_demo_sk              = customer.c_current_hdemo_sk
and     customer_address.ca_address_sk           = customer.c_current_addr_sk
and     date_dim.d_year                  = 1999 
and     date_dim.d_moy                   = 11
and     ( (customer_demographics.cd_marital_status       = 'M' and customer_demographics.cd_education_status     = 'Unknown')
        or(customer_demographics.cd_marital_status       = 'W' and customer_demographics.cd_education_status     = 'Advanced Degree'))
and     household_demographics.hd_buy_potential like '>10000%'
and     customer_address.ca_gmt_offset           = -7
group by call_center.cc_call_center_id,call_center.cc_name,call_center.cc_manager,cd_marital_status,cd_education_status
order by sum(catalog_returns.cr_net_loss) desc;

