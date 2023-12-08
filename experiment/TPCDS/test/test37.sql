-- start query 1 in stream 5 using template query37.tpl
select  i_item_id
       ,i_item_desc
       ,i_current_price
 from item, inventory, date_dim, catalog_sales
 where item.i_current_price between 12 and 12 + 30
 and inventory.inv_item_sk = item.i_item_sk
 and date_dim.d_date_sk=inventory.inv_date_sk
 and date_dim.d_date between cast('2000-07-18' as date) and (cast('2000-07-18' as date) + '60 days'::interval)
 and item.i_manufact_id in (718,771,966,937)
 and inventory.inv_quantity_on_hand between 100 and 500
 and catalog_sales.cs_item_sk = item.i_item_sk
 group by i_item_id,i_item_desc,i_current_price
 order by i_item_id
 limit 100;

