-- start query 1 in stream 0 using template query20.tpl
select  i_item_id
       ,i_item_desc 
       ,i_category 
       ,i_class 
       ,i_current_price
       ,sum(cs_ext_sales_price) as itemrevenue 
       ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
           (partition by i_class) as revenueratio
 from	catalog_sales
     ,item 
     ,date_dim
 where catalog_sales.cs_item_sk = item.i_item_sk 
   and item.i_category in ('Jewelry', 'Sports', 'Books')
   and catalog_sales.cs_sold_date_sk = date_dim.d_date_sk
 and date_dim.d_date between cast('2001-01-12' as date) and (cast('2001-01-12' as date) + '30 days'::interval)
 group by i_item_id
         ,i_item_desc 
         ,i_category
         ,i_class
         ,i_current_price
 order by i_category
         ,i_class
         ,i_item_id
         ,i_item_desc
         ,revenueratio
limit 100;

