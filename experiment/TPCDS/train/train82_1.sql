-- start query 1 in stream 0 using template query82.tpl
select  i_item_id
       ,i_item_desc
       ,i_current_price
 from item, inventory, date_dim, store_sales
 where item.i_current_price between 30 and 30+30
 and inventory.inv_item_sk = item.i_item_sk
 and date_dim.d_date_sk=inventory.inv_date_sk
 and date_dim.d_date between cast('2002-05-30' as date) and (cast('2002-05-30' as date) + '60 days'::interval)
 and item.i_manufact_id in (437,129,727,663)
 and inventory.inv_quantity_on_hand between 100 and 500
 and store_sales.ss_item_sk = item.i_item_sk
 group by item.i_item_id,item.i_item_desc,item.i_current_price
 order by item.i_item_id
 limit 100;

