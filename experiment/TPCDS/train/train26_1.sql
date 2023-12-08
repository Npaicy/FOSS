-- start query 1 in stream 0 using template query26.tpl
select  i_item_id, 
        avg(cs_quantity) agg1,
        avg(cs_list_price) agg2,
        avg(cs_coupon_amt) agg3,
        avg(cs_sales_price) agg4 
 from catalog_sales, customer_demographics, date_dim, item, promotion
 where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk and
       catalog_sales.cs_item_sk = item.i_item_sk and
       catalog_sales.cs_bill_cdemo_sk = customer_demographics.cd_demo_sk and
       catalog_sales.cs_promo_sk = promotion.p_promo_sk and
       customer_demographics.cd_gender = 'F' and 
       customer_demographics.cd_marital_status = 'W' and
       customer_demographics.cd_education_status = 'Primary' and
       (promotion.p_channel_email = 'N' or promotion.p_channel_event = 'N') and
       date_dim.d_year = 1998 
 group by i_item_id
 order by i_item_id
 limit 100;

