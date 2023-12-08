-- start query 1 in stream 1 using template query96.tpl
select  count(*) 
from store_sales
    ,household_demographics 
    ,time_dim, store
where store_sales.ss_sold_time_sk = time_dim.t_time_sk   
    and store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk 
    and store_sales.ss_store_sk = store.s_store_sk
    and time_dim.t_hour = 20
    and time_dim.t_minute >= 30
    and household_demographics.hd_dep_count = 8
    and store.s_store_name = 'ese'
order by count(*)
limit 100;

