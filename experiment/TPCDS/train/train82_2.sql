-- start query 1 in stream 1 using template query82.tpl
select  i_item_id
       ,i_item_desc
       ,i_current_price
 from item, inventory, date_dim, store_sales
 where item.i_current_price between 3 and 3+30
 and inventory.inv_item_sk = item.i_item_sk
 and date_dim.d_date_sk=inventory.inv_date_sk
 and date_dim.d_date between cast('1998-05-20' as date) and (cast('1998-05-20' as date) + '60 days'::interval)
 and item.i_manufact_id in (59,526,301,399)
 and inventory.inv_quantity_on_hand between 100 and 500
 and store_sales.ss_item_sk = item.i_item_sk
 group by item.i_item_id,item.i_item_desc,item.i_current_price
 order by item.i_item_id
 limit 100;

