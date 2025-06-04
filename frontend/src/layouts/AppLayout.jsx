// frontend/src/layouts/AppLayout.jsx - MODIFICADO
import React, { useState, useEffect } from 'react';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Container,
  useMediaQuery,
  useTheme,
  Collapse,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import SettingsIcon from '@mui/icons-material/Settings';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import PaymentIcon from '@mui/icons-material/Payment';
import HistoryIcon from '@mui/icons-material/History';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ViewListIcon from '@mui/icons-material/ViewList';
import logoImg from '../assets/agentia2.png';
import { useQueryStore } from '../contexts/store';

const AppLayout = ({ children }) => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  
  // Get recent queries from your store
  const { recentQueries } = useQueryStore();
  
  const toggleDrawer = (open) => (event) => {
    if (
      event.type === 'keydown' &&
      (event.key === 'Tab' || event.key === 'Shift')
    ) {
      return;
    }
    setDrawerOpen(open);
  };
  
  const toggleHistory = () => {
    setHistoryOpen(!historyOpen);
  };

  // Determine if a sidebar item is active based on current route
  const isActive = (path) => {
    return location.pathname === path;
  };

  // Sidebar menu items
  const sidebarItems = [
    { 
      text: 'Consultas', 
      icon: <QuestionAnswerIcon />, 
      path: '/',
      active: isActive('/')
    },
    { 
      text: 'Novedades regulatorias', 
      icon: <NewReleasesIcon />, 
      path: '/novedades',
      active: isActive('/novedades')
    },
    { 
      text: 'Historial', 
      icon: <HistoryIcon />, 
      onClick: toggleHistory,
      expandable: true,
      expanded: historyOpen,
      // Add a "View All" option and then the recent queries
      subItems: [
        {
          text: 'Ver todo el historial',
          path: '/historial',
          icon: <ViewListIcon />,
          active: isActive('/historial'),
          isViewAll: true
        },
        ...recentQueries?.slice(0, 5).map((query, index) => ({
          text: query.text?.length > 30 ? `${query.text.substring(0, 30)}...` : query.text || `Consulta ${index + 1}`,
          path: `/consulta/${query.id}`,
          timestamp: query.timestamp,
          active: location.pathname === `/consulta/${query.id}`,
          type: query.type
        })) || []
      ]
    }
  ];

  // Function to get type icon for query
  const getQueryTypeIcon = (type) => {
    switch (type) {
      case 'gap_analysis':
        return '📊';
      case 'consultation':
        return '💬';
      default:
        return '❓';
    }
  };

  // The sidebar component
  const sidebar = (
    <Box
      sx={{ 
        width: 280,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid rgba(0, 0, 0, 0.12)'
      }}
      role="presentation"
      onClick={isMobile ? toggleDrawer(false) : undefined}
      onKeyDown={isMobile ? toggleDrawer(false) : undefined}
    >
      {/* AJUSTE: Reducir padding superior para compensar header más pequeño */}
      <Box sx={{ pt: 6 }}></Box> {/* Reducido de pt: 10 a pt: 6 */}
      <Box sx={{ flexGrow: 1 }}>
        <List component="nav" sx={{ p: 1, pt: 2 }}> {/* Reducido de pt: 4 a pt: 2 */}
          {sidebarItems.map((item, index) => (
            <React.Fragment key={item.text}>
              <ListItem 
                button 
                component={item.expandable ? 'div' : RouterLink} 
                to={item.expandable ? undefined : item.path}
                onClick={item.onClick}
                sx={{
                  borderRadius: '8px',
                  mb: 0.5,
                  backgroundColor: item.active ? 'rgba(77, 10, 46, 0.08)' : 'transparent',
                  color: item.active ? 'primary.main' : 'text.primary',
                  '&:hover': {
                    backgroundColor: 'rgba(77, 10, 46, 0.04)',
                  }
                }}
              >
                <ListItemIcon sx={{ 
                  color: item.active ? 'primary.main' : 'inherit',
                  minWidth: '40px'
                }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text} 
                  primaryTypographyProps={{ 
                    fontWeight: item.active ? 500 : 400 
                  }}
                />
                {item.expandable && (
                  item.expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />
                )}
              </ListItem>
              
              {item.expandable && (
                <Collapse in={item.expanded} timeout="auto" unmountOnExit>
                  <List component="div" disablePadding>
                    {item.subItems.length > 0 ? (
                      item.subItems.map((subItem, subIndex) => (
                        <ListItem
                          key={subIndex}
                          button
                          component={RouterLink}
                          to={subItem.path}
                          sx={{
                            pl: subItem.isViewAll ? 4 : 5,
                            py: subItem.isViewAll ? 1 : 0.5,
                            borderRadius: '8px',
                            backgroundColor: subItem.active ? 'rgba(77, 10, 46, 0.08)' : 'transparent',
                            color: subItem.active ? 'primary.main' : 'text.secondary',
                            '&:hover': {
                              backgroundColor: 'rgba(77, 10, 46, 0.04)',
                            },
                            // Special styling for "View All" option
                            ...(subItem.isViewAll && {
                              borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
                              mb: 1,
                              fontWeight: 500
                            })
                          }}
                        >
                          {subItem.isViewAll && (
                            <ListItemIcon sx={{ minWidth: '32px' }}>
                              {subItem.icon}
                            </ListItemIcon>
                          )}
                          <ListItemText 
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                {!subItem.isViewAll && subItem.type && (
                                  <Typography component="span" sx={{ fontSize: '0.8rem' }}>
                                    {getQueryTypeIcon(subItem.type)}
                                  </Typography>
                                )}
                                <Typography
                                  component="span"
                                  variant={subItem.isViewAll ? 'body2' : 'body2'}
                                  sx={{ 
                                    fontWeight: subItem.isViewAll ? 500 : 400,
                                    fontSize: subItem.isViewAll ? '0.875rem' : '0.8rem'
                                  }}
                                >
                                  {subItem.text}
                                </Typography>
                              </Box>
                            }
                            secondary={!subItem.isViewAll && subItem.timestamp ? new Date(subItem.timestamp).toLocaleDateString() : null}
                            primaryTypographyProps={{ 
                              fontWeight: subItem.active ? 500 : 400
                            }}
                            secondaryTypographyProps={{
                              variant: 'caption',
                              fontSize: '0.7rem'
                            }}
                          />
                        </ListItem>
                      ))
                    ) : (
                      <ListItem sx={{ pl: 4, py: 1 }}>
                        <ListItemText 
                          primary="No hay consultas recientes" 
                          primaryTypographyProps={{ 
                            variant: 'body2', 
                            color: 'text.secondary',
                            fontStyle: 'italic'
                          }} 
                        />
                      </ListItem>
                    )}
                  </List>
                </Collapse>
              )}
            </React.Fragment>
          ))}
        </List>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* MODIFICACIÓN PRINCIPAL: AppBar más estrecho y sin menú superior derecho */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          // Personalizar altura del AppBar
          '& .MuiToolbar-root': {
            minHeight: '48px !important', // Reducido de 64px a 48px
            height: '48px',
            paddingTop: '8px',
            paddingBottom: '8px'
          }
        }}
      >
        <Toolbar sx={{ minHeight: '48px !important', height: '48px' }}>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleDrawer(true)}
              sx={{ mr: 2 }}
              size="small" // Botón más pequeño
            >
              <MenuIcon />
            </IconButton>
          )}
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <img 
              src={logoImg}
              alt="MINSAIT Logo" 
              style={{ 
                height: '70px', // Reducido de 65px a 32px
                marginRight: '2px',
                marginTop: '5px'
              }} 
            />
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          {/* ELIMINADO COMPLETAMENTE: El menú superior derecho */}
          {/* 
          {!isMobile && (
            <Box>
              <Button color="inherit" component={RouterLink} to="/">
                Inicio
              </Button>
              <Button color="inherit" component={RouterLink} to="/">
                Consultas
              </Button>
              <Button color="inherit" component={RouterLink} to="/">
                Ayuda
              </Button>
            </Box>
          )}
          */}
        </Toolbar>
      </AppBar>

      {/* Mobile drawer */}
      <Drawer
        anchor="left"
        open={isMobile && drawerOpen}
        onClose={toggleDrawer(false)}
        sx={{
          '& .MuiDrawer-paper': { 
            width: 280,
            boxSizing: 'border-box' 
          },
        }}
      >
        {sidebar}
      </Drawer>

      {/* Permanent sidebar for desktop */}
      {!isMobile && (
        <Drawer
          variant="permanent"
          sx={{
            width: 280,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { 
              width: 280, 
              boxSizing: 'border-box',
              borderRight: '1px solid rgba(0, 0, 0, 0.12)',
              zIndex: (theme) => theme.zIndex.drawer,
              top: '48px', // AJUSTADO: de 64px a 48px
              height: 'calc(100% - 48px)', // AJUSTADO: de 64px a 48px
              overflowY: 'visible'
            },
          }}
          open
        >
          {sidebar}
        </Drawer>
      )}

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          bgcolor: 'background.default',
          p: 3,
          mt: '48px', // AJUSTADO: de 64px a 48px (altura del AppBar)
          ml: isMobile ? 0 : '280px', // Ajustar para el ancho del sidebar
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default AppLayout;