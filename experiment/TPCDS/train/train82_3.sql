-- start query 1 in stream 2 using template query82.tpl
select  i_item_id
       ,i_item_desc
       ,i_current_price
 from item, inventory, date_dim, store_sales
 where item.i_current_price between 34 and 34+30
 and inventory.inv_item_sk = item.i_item_sk
 and date_dim.d_date_sk=inventory.inv_date_sk
 and date_dim.d_date between cast('1999-01-24' as date) and (cast('1999-01-24' as date) + '60 days'::interval)
 and item.i_manufact_id in (33,652,78,269)
 and inventory.inv_quantity_on_hand between 100 and 500
 and store_sales.ss_item_sk = item.i_item_sk
 group by item.i_item_id,item.i_item_desc,item.i_current_price
 order by item.i_item_id
 limit 100;

