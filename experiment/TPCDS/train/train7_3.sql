-- start query 1 in stream 2 using template query7.tpl
select  i_item_id, 
        avg(ss_quantity) agg1,
        avg(ss_list_price) agg2,
        avg(ss_coupon_amt) agg3,
        avg(ss_sales_price) agg4 
 from store_sales, customer_demographics, date_dim, item, promotion
 where store_sales.ss_sold_date_sk = date_dim.d_date_sk and
       store_sales.ss_item_sk = item.i_item_sk and
       store_sales.ss_cdemo_sk = customer_demographics.cd_demo_sk and
       store_sales.ss_promo_sk = promotion.p_promo_sk and
       customer_demographics.cd_gender = 'M' and 
       customer_demographics.cd_marital_status = 'W' and
       customer_demographics.cd_education_status = 'College' and
       (promotion.p_channel_email = 'N' or promotion.p_channel_event = 'N') and
       date_dim.d_year = 2001 
 group by i_item_id
 order by i_item_id
 limit 100;

