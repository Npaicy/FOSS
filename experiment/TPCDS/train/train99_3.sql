-- start query 1 in stream 2 using template query99.tpl
select  
   substr(w_warehouse_name,1,20)
  ,sm_type
  ,cc_name
  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk <= 30 ) then 1 else 0 end)  as "30 days" 
  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 30) and 
                 (cs_ship_date_sk - cs_sold_date_sk <= 60) then 1 else 0 end )  as "31-60 days" 
  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 60) and 
                 (cs_ship_date_sk - cs_sold_date_sk <= 90) then 1 else 0 end)  as "61-90 days" 
  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk > 90) and
                 (cs_ship_date_sk - cs_sold_date_sk <= 120) then 1 else 0 end)  as "91-120 days" 
  ,sum(case when (cs_ship_date_sk - cs_sold_date_sk  > 120) then 1 else 0 end)  as ">120 days" 
from
   catalog_sales
  ,warehouse
  ,ship_mode
  ,call_center
  ,date_dim
where
    date_dim.d_month_seq between 1178 and 1178 + 11
and catalog_sales.cs_ship_date_sk   = date_dim.d_date_sk
and catalog_sales.cs_warehouse_sk   = warehouse.w_warehouse_sk
and catalog_sales.cs_ship_mode_sk   = ship_mode.sm_ship_mode_sk
and catalog_sales.cs_call_center_sk = call_center.cc_call_center_sk
group by
   substr(warehouse.w_warehouse_name,1,20)
  ,ship_mode.sm_type
  ,call_center.cc_name
order by substr(warehouse.w_warehouse_name,1,20)
        ,ship_mode.sm_type
        ,call_center.cc_name
limit 100;

