install.packages("bupaR")
install.packages("edeaR")
install.packages("eventdataR")
install.packages("processmapR")
install.packages("processmonitR")
install.packages("xesreadR")
install.packages("petrinetR")
library(bupaR)

getwd()

read.csv('/home/workspace/files/buck_project/abhijeet/results/event_logs.csv')
event_logs <- read.csv('/home/workspace/files/buck_project/abhijeet/results/event_logs.csv')
event_logs <- event_logs %>% mutate(timestamp=as.Date(timestamp, format = "%Y-%m-%d"))
View(event_logs)

event_logs_bupar <- event_logs %>%
  eventlog(
    case_id = "UPN",
    activity_id = "activity_id",
    activity_instance_id = "activity_instance",
    lifecycle_id = "status",
    timestamp = "timestamp",
    resource_id = "resource"
  )
View(event_logs_bupar)

event_logs_bupar %>% processing_time("activity", units = "days") %>% plot()
event_logs_bupar %>% activity_presence() %>% plot
# event_logs_bupar %>% process_map(type=frequency('relative'))
event_logs_bupar %>% precedence_matrix(type = "absolute") %>% plot

event_logs_bupar %>% trace_explorer(coverage=0.99, type='frequent')
event_logs_bupar %>% trace_explorer(coverage=0.001, type='infrequent')
